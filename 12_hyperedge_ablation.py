#!/usr/bin/env python
"""
12_hyperedge_ablation.py — 4-hyperedge 개별 ablation.

평가자 핵심 비판 대응:
    "4종 hyperedge 통합이 핵심 architectural claim인데, 각 hyperedge를
     빼본 ablation이 없으면 '왜 4종인가, 왜 이 조합인가'에 답할 수가 없음"

실행 조건:
    - fold 0, seed 42 (단일 fold/seed, 평가자 권장)
    - B6 (4-edge K-AHFM-Clinic)
    - 5 conditions: full + 4 single ablations
    - 2 scenarios: Early + Full
    - 총 10 runs (각 ~2-3분, 총 ~30분)

Ablation 방식:
    Edge masking at batch level — forward 시점에 batch['edge_mask']에서
    excluded type의 edge들을 False로 설정. 모델 코드 변경 불필요.

산출물:
    <output_dir>/
        ablation_runs.jsonl              # 10 runs raw results
        ablation_summary.json             # 집계
        ablation_summary.md               # 표 형식

사용법:
    python 12_hyperedge_ablation.py \\
        --extracted <repo>/data/extracted \\
        --processed <repo>/data/processed \\
        --embeddings <repo>/data/embeddings \\
        --output <repo>/results/hyperedge_ablation
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# Edge type names (B6 4 types only)
EDGE_TYPE_NAMES = {
    0: 'E_trajectory',
    1: 'E_co-trajectory',
    2: 'E_intervention-response',
    3: 'E_clinical_prior',
}


def mask_edges_by_type(batch, excluded_type):
    """배치의 edge_mask에서 excluded_type을 가진 edge들을 False로 설정.

    Args:
        batch: dict containing 'edge_types' (B, E) and 'edge_mask' (B, E)
        excluded_type: None (no exclusion) or int (0-3)

    Returns:
        batch with modified edge_mask
    """
    if excluded_type is None:
        return batch
    edge_types = batch['edge_types']
    edge_mask = batch['edge_mask']
    # 해당 type 가진 edge를 False로
    type_match = (edge_types == excluded_type)
    new_mask = edge_mask & (~type_match)
    batch['edge_mask'] = new_mask
    return batch


def to_device(batch, device, excluded_type=None):
    """배치를 device로 옮기고 + ablation mask 적용."""
    for k in ('node_features', 'node_mask', 'severity_first_3', 'severity_mean_input',
              'tr_label', 'tr_valid', 'diag_label'):
        if k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
    for k in ('adjacency', 'incidence', 'edge_mask', 'edge_types'):
        if k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
    # Apply ablation mask AFTER moving to device
    batch = mask_edges_by_type(batch, excluded_type)
    return batch


def load_data(args):
    from data.parser import parse_all_sessions
    from data.labels import load_labels_json
    from data.splits import SplitResult

    print(f"  Sessions 파싱: {args.extracted}/training/labeling ...")
    labeling = Path(args.extracted) / 'training' / 'labeling'
    sessions = parse_all_sessions(labeling, progress=False)
    by_patient = defaultdict(list)
    for s in sessions:
        if s.patient_id and s.session_number > 0:
            by_patient[s.patient_id].append(s)
    for pid in by_patient:
        by_patient[pid].sort(key=lambda s: s.session_number)
    print(f"  → {len(sessions):,} 세션 / {len(by_patient):,} 환자")

    patient_labels = load_labels_json(Path(args.processed) / 'patient_labels.json')
    print(f"  환자 라벨: {len(patient_labels):,}명")

    fold_tr = SplitResult.from_json(Path(args.processed) / 'fold_tr.json')
    print(f"  Fold TR: {fold_tr.n_patients}명, {fold_tr.n_folds}-fold")

    return dict(by_patient), patient_labels, fold_tr


def build_graphs_cached(by_patient, patient_labels, embeddings_dir, scenario,
                        input_window, _cache=None):
    """동일 scenario는 재사용. _cache로 fold-loop 안에서 그래프 재빌드 회피."""
    from graph.build_graph import build_all_patient_graphs

    graphs = build_all_patient_graphs(
        sessions_by_patient=by_patient,
        patient_labels=patient_labels,
        embeddings_cache_dir=Path(embeddings_dir),
        model_type='B6',
        scenario=scenario,
        input_window=input_window,
        progress=False,
    )
    return graphs


def run_single_ablation(
    train_graphs, val_graphs, test_graphs, diag_weights,
    config, device, scenario, fold, seed, excluded_type,
    verbose=False,
):
    """ablation을 적용한 1 condition × 1 scenario 학습 + 평가."""
    from models.graph_models import build_graph_model, multitask_loss
    from training.graph_trainer import (
        PatientGraphDataset, make_collate_fn, evaluate as _eval_orig,
    )
    from training.metrics import compute_diag_metrics, compute_per_diagnosis_tr, compute_tr_metrics

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = build_graph_model(
        'B6', hidden_dim=config['hidden_dim'], dropout=config['dropout'],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
    )

    collate = make_collate_fn('B6')
    train_ds = PatientGraphDataset(train_graphs)
    val_ds = PatientGraphDataset(val_graphs) if val_graphs else None
    test_ds = PatientGraphDataset(test_graphs)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'] * 2, shuffle=False,
                            collate_fn=collate) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'] * 2, shuffle=False,
                             collate_fn=collate)

    diag_weights = diag_weights.to(device) if diag_weights is not None else None

    # Custom evaluate with ablation masking
    @torch.no_grad()
    def evaluate_ablated(model, loader, full_metrics=False):
        model.eval()
        all_tr_logits, all_tr_labels, all_tr_valid = [], [], []
        all_diag_logits, all_diag_labels = [], []

        for batch in loader:
            batch = to_device(batch, device, excluded_type=excluded_type)
            outputs = model(batch)
            all_tr_logits.append(outputs['tr_logit'].cpu())
            all_tr_labels.append(batch['tr_label'].cpu())
            all_tr_valid.append(batch['tr_valid'].cpu())
            all_diag_logits.append(outputs['diag_logits'].cpu())
            all_diag_labels.append(batch['diag_label'].cpu())

        tr_logits = torch.cat(all_tr_logits)
        tr_labels = torch.cat(all_tr_labels).numpy()
        tr_valid = torch.cat(all_tr_valid).numpy().astype(bool)
        diag_logits = torch.cat(all_diag_logits)
        diag_labels = torch.cat(all_diag_labels).numpy()

        tr_probs = torch.sigmoid(tr_logits).numpy()
        diag_probs = torch.softmax(diag_logits, dim=-1).numpy()
        diag_preds = diag_logits.argmax(dim=-1).numpy()

        if tr_valid.sum() > 0:
            tr_m = compute_tr_metrics(tr_labels[tr_valid], tr_probs[tr_valid])
        else:
            tr_m = compute_tr_metrics(np.array([]), np.array([]))

        diag_m = compute_diag_metrics(diag_labels, diag_preds, diag_probs)
        result = {'tr': tr_m, 'diag': diag_m}
        if full_metrics:
            result['per_diag_tr'] = compute_per_diagnosis_tr(
                tr_labels, tr_probs, diag_labels, tr_valid,
            )
        return result

    best_val_auroc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience = config['patience']

    start = time.time()
    for epoch in range(config['max_epochs']):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = to_device(batch, device, excluded_type=excluded_type)
            optimizer.zero_grad()
            outputs = model(batch)
            loss, _ = multitask_loss(
                outputs, batch,
                lambda_tr=config['lambda_tr'],
                lambda_diag=config['lambda_diag'],
                focal_alpha=config['focal_alpha'],
                focal_gamma=config['focal_gamma'],
                diag_class_weights=diag_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        if val_loader is not None:
            val_m = evaluate_ablated(model, val_loader)
            cur_auroc = val_m['tr']['auroc']
        else:
            val_m = None
            cur_auroc = 0.0

        if verbose and (epoch + 1) % 10 == 0:
            print(f"      ep {epoch+1}: train_loss {train_loss_sum/max(n_batches,1):.4f}  val_auroc {cur_auroc:.4f}")

        if val_loader is not None and cur_auroc > best_val_auroc:
            best_val_auroc = cur_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = evaluate_ablated(model, test_loader, full_metrics=True)
    train_time = time.time() - start

    return {
        'condition': 'full_B6' if excluded_type is None else f'minus_{EDGE_TYPE_NAMES[excluded_type]}',
        'excluded_type': excluded_type,
        'excluded_name': None if excluded_type is None else EDGE_TYPE_NAMES[excluded_type],
        'scenario': scenario,
        'fold': fold,
        'seed': seed,
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch,
        'epochs_trained': epoch + 1,
        'train_time_sec': round(train_time, 1),
        'test_tr': test_m['tr'],
        'test_diag': test_m['diag'],
        'test_per_diag_tr': test_m['per_diag_tr'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted', required=True, type=str)
    parser.add_argument('--processed', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input-window', type=int, default=3)

    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lambda-tr', type=float, default=1.0)
    parser.add_argument('--lambda-diag', type=float, default=0.3)
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'batch_size': args.batch_size, 'max_epochs': args.max_epochs,
        'patience': args.patience,
        'lambda_tr': args.lambda_tr, 'lambda_diag': args.lambda_diag,
        'focal_alpha': args.focal_alpha, 'focal_gamma': args.focal_gamma,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  4-Hyperedge 개별 Ablation")
    print(f"{'='*70}")
    print(f"  Device:   {device}")
    print(f"  Fold/Seed: {args.fold}/{args.seed}")
    print(f"  Output:   {output_dir}")
    print(f"  Conditions:")
    print(f"    1. full_B6 (4-edge)                  — baseline")
    print(f"    2. minus_E_trajectory               (excluded_type=0)")
    print(f"    3. minus_E_co-trajectory            (excluded_type=1)")
    print(f"    4. minus_E_intervention-response    (excluded_type=2)")
    print(f"    5. minus_E_clinical_prior           (excluded_type=3)")
    print(f"  Scenarios: Early + Full = 10 runs total\n")

    by_patient, patient_labels, fold_tr = load_data(args)
    fold_info = next(f for f in fold_tr.folds if f.fold_index == args.fold)
    train_pids = set(fold_info.train_patients)
    test_pids = set(fold_info.test_patients)

    runs_path = output_dir / 'ablation_runs.jsonl'
    runs_file = open(runs_path, 'w', encoding='utf-8')

    all_runs = []
    total_runs = 5 * 2
    run_idx = 0
    overall_start = time.time()

    for scenario in ['early', 'full']:
        print(f"\n--- Scenario: {scenario} ---")
        print(f"  그래프 빌드 (B6, {scenario}) ...")
        graphs = build_graphs_cached(
            by_patient, patient_labels, args.embeddings, scenario, args.input_window,
        )
        n_tr_valid = sum(1 for g in graphs.values() if g.tr_valid)
        print(f"  → {len(graphs):,} 환자 / TR-valid {n_tr_valid}명")

        train_graphs = sorted(
            [g for pid, g in graphs.items() if pid in train_pids],
            key=lambda g: g.patient_id,
        )
        test_graphs = sorted(
            [g for pid, g in graphs.items() if pid in test_pids],
            key=lambda g: g.patient_id,
        )
        n_val = max(1, int(len(train_graphs) * 0.1))
        val_graphs = train_graphs[:n_val]
        train_graphs_actual = train_graphs[n_val:]
        print(f"  Split: train {len(train_graphs_actual)}, val {len(val_graphs)}, test {len(test_graphs)}")

        from training.graph_trainer import compute_diagnosis_class_weights
        diag_weights = compute_diagnosis_class_weights(train_graphs_actual)

        # 5 conditions: full + 4 ablations
        conditions = [
            (None, 'full_B6 (4-edge)'),
            (0, 'minus E_trajectory'),
            (1, 'minus E_co-trajectory'),
            (2, 'minus E_intervention-response'),
            (3, 'minus E_clinical_prior'),
        ]

        for excluded_type, cond_name in conditions:
            run_idx += 1
            start = time.time()
            print(f"\n  [{run_idx}/{total_runs}] {cond_name} ({scenario})", flush=True)
            result = run_single_ablation(
                train_graphs_actual, val_graphs, test_graphs,
                diag_weights, config, device,
                scenario, args.fold, args.seed, excluded_type,
                verbose=False,
            )
            elapsed = time.time() - start

            runs_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            runs_file.flush()
            all_runs.append(result)

            tr_auroc = result['test_tr']['auroc']
            tr_f1 = result['test_tr']['f1_pos']
            diag_f1 = result['test_diag']['macro_f1']
            print(f"      → val_auroc {result['best_val_auroc']:.3f}  "
                  f"test_tr_auroc {tr_auroc:.3f}  test_tr_f1 {tr_f1:.3f}  "
                  f"test_diag_f1 {diag_f1:.3f}  ({elapsed:.1f}s)")

        del graphs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    runs_file.close()
    overall_time = time.time() - overall_start

    # Aggregate & Summary
    print(f"\n{'='*70}")
    print(f"  결과 집계 (총 {overall_time:.0f}초 = {overall_time/60:.1f}분)")
    print(f"{'='*70}\n")

    # Build summary tables
    by_scenario = {'early': {}, 'full': {}}
    for r in all_runs:
        by_scenario[r['scenario']][r['condition']] = r

    summary = {
        'fold': args.fold,
        'seed': args.seed,
        'total_time_sec': round(overall_time, 1),
        'by_scenario': by_scenario,
    }
    with open(output_dir / 'ablation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ ablation_summary.json saved")

    # Markdown summary
    md_lines = [
        f"# 4-Hyperedge Ablation 결과 (fold {args.fold}, seed {args.seed})\n",
        f"\nTotal time: {overall_time:.0f}s ({overall_time/60:.1f}min)\n",
    ]

    for scenario in ['early', 'full']:
        md_lines.append(f"\n## Scenario: {scenario}\n")
        md_lines.append("\n### TR (binary classification)\n")
        md_lines.append("| Condition | TR AUROC | TR F1_pos | TR MCC | Bal Acc |")
        md_lines.append("|---|---|---|---|---|")
        # Baseline first
        full_run = by_scenario[scenario].get('full_B6 (4-edge)')
        if full_run:
            t = full_run['test_tr']
            md_lines.append(f"| **full B6 (baseline)** | {t['auroc']:.3f} | {t['f1_pos']:.3f} | {t['mcc']:.3f} | {t['balanced_acc']:.3f} |")
        # Then ablations
        for et in range(4):
            cond_key = f'minus_{EDGE_TYPE_NAMES[et]}'
            run = by_scenario[scenario].get(cond_key)
            if run:
                t = run['test_tr']
                # Compute deltas from baseline
                if full_run:
                    delta_auroc = t['auroc'] - full_run['test_tr']['auroc']
                    delta_str = f" ({delta_auroc:+.3f})"
                else:
                    delta_str = ""
                md_lines.append(f"| − {EDGE_TYPE_NAMES[et]} | {t['auroc']:.3f}{delta_str} | {t['f1_pos']:.3f} | {t['mcc']:.3f} | {t['balanced_acc']:.3f} |")

        md_lines.append("\n### Diagnosis (4-class)\n")
        md_lines.append("| Condition | Diag macro F1 | Diag Accuracy |")
        md_lines.append("|---|---|---|")
        if full_run:
            d = full_run['test_diag']
            md_lines.append(f"| **full B6 (baseline)** | {d['macro_f1']:.3f} | {d['accuracy']:.3f} |")
        for et in range(4):
            cond_key = f'minus_{EDGE_TYPE_NAMES[et]}'
            run = by_scenario[scenario].get(cond_key)
            if run:
                d = run['test_diag']
                if full_run:
                    delta_f1 = d['macro_f1'] - full_run['test_diag']['macro_f1']
                    delta_str = f" ({delta_f1:+.3f})"
                else:
                    delta_str = ""
                md_lines.append(f"| − {EDGE_TYPE_NAMES[et]} | {d['macro_f1']:.3f}{delta_str} | {d['accuracy']:.3f} |")

    summary_md = '\n'.join(md_lines)
    with open(output_dir / 'ablation_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"  ✓ ablation_summary.md saved")
    print("\n" + summary_md)


if __name__ == '__main__':
    main()
