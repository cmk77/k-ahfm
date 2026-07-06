#!/usr/bin/env python
"""
12_hyperedge_ablation_v2.py — 5-fold × 5-seed 전체 4-hyperedge ablation.

v1 대비 변경:
    - 5-fold × 5-seed 평가 (250 runs total)
    - 95% CI 집계 보고
    - §4.1 표 3a/3b와 동일한 통계 체계
    - 매 run jsonl로 즉시 flush (interrupt 안전)

실행 조건:
    - 5 conditions: full B6 + 4 single ablations
    - 5 folds × 5 seeds = 25 runs per condition per scenario
    - 2 scenarios: Early + Full
    - 총 250 runs (각 ~4-5초, 총 ~20분)

산출물:
    <output_dir>/
        ablation_v2_runs.jsonl              # 250 raw runs
        ablation_v2_aggregated.json          # 5 conditions × 2 scenarios 집계 (mean ± 95% CI)
        ablation_v2_summary.md               # 표 형식 (논문용)

사용법:
    python 12_hyperedge_ablation_v2.py \\
        --extracted <repo>/data/extracted \\
        --processed <repo>/data/processed \\
        --embeddings <repo>/data/embeddings \\
        --output <repo>/results/hyperedge_ablation_v2
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


EDGE_TYPE_NAMES = {
    0: 'E_trajectory',
    1: 'E_co-trajectory',
    2: 'E_intervention-response',
    3: 'E_clinical_prior',
}

# Seeds same as 09_train_graph_models.py
DEFAULT_SEEDS = [0, 42, 2026, 7, 1024]


def mask_edges_by_type(batch, excluded_type):
    if excluded_type is None:
        return batch
    edge_types = batch['edge_types']
    edge_mask = batch['edge_mask']
    type_match = (edge_types == excluded_type)
    new_mask = edge_mask & (~type_match)
    batch['edge_mask'] = new_mask
    return batch


def to_device(batch, device, excluded_type=None):
    for k in ('node_features', 'node_mask', 'severity_first_3', 'severity_mean_input',
              'tr_label', 'tr_valid', 'diag_label'):
        if k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
    for k in ('adjacency', 'incidence', 'edge_mask', 'edge_types'):
        if k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
    batch = mask_edges_by_type(batch, excluded_type)
    return batch


def load_data(args):
    from data.parser import parse_all_sessions
    from data.labels import load_labels_json
    from data.splits import SplitResult

    print(f"  Sessions 파싱 ...")
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
    fold_tr = SplitResult.from_json(Path(args.processed) / 'fold_tr.json')
    print(f"  Fold TR: {fold_tr.n_patients}명, {fold_tr.n_folds}-fold\n")

    return dict(by_patient), patient_labels, fold_tr


def build_graphs(by_patient, patient_labels, embeddings_dir, scenario, input_window):
    from graph.build_graph import build_all_patient_graphs
    return build_all_patient_graphs(
        sessions_by_patient=by_patient,
        patient_labels=patient_labels,
        embeddings_cache_dir=Path(embeddings_dir),
        model_type='B6',
        scenario=scenario,
        input_window=input_window,
        progress=False,
    )


def run_single(
    train_graphs, val_graphs, test_graphs, diag_weights,
    config, device, scenario, fold, seed, excluded_type,
):
    from models.graph_models import build_graph_model, multitask_loss
    from training.graph_trainer import PatientGraphDataset, make_collate_fn
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

    @torch.no_grad()
    def evaluate_ablated(loader, full_metrics=False):
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
        for batch in train_loader:
            batch = to_device(batch, device, excluded_type=excluded_type)
            optimizer.zero_grad()
            outputs = model(batch)
            loss, _ = multitask_loss(
                outputs, batch,
                lambda_tr=config['lambda_tr'], lambda_diag=config['lambda_diag'],
                focal_alpha=config['focal_alpha'], focal_gamma=config['focal_gamma'],
                diag_class_weights=diag_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_loader is not None:
            cur_auroc = evaluate_ablated(val_loader)['tr']['auroc']
        else:
            cur_auroc = 0.0

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

    test_m = evaluate_ablated(test_loader, full_metrics=True)
    train_time = time.time() - start

    return {
        'condition': 'full_B6' if excluded_type is None else f'minus_{EDGE_TYPE_NAMES[excluded_type]}',
        'excluded_type': excluded_type,
        'excluded_name': None if excluded_type is None else EDGE_TYPE_NAMES[excluded_type],
        'scenario': scenario, 'fold': fold, 'seed': seed,
        'best_val_auroc': float(best_val_auroc),
        'best_epoch': int(best_epoch),
        'epochs_trained': int(epoch + 1),
        'train_time_sec': round(train_time, 2),
        'test_tr': test_m['tr'],
        'test_diag': test_m['diag'],
        'test_per_diag_tr': test_m['per_diag_tr'],
    }


def aggregate_metrics_with_ci(values):
    """Mean + 95% CI (using 1.96 * SD / sqrt(N))."""
    if not values:
        return {'mean': 0, 'std': 0, 'ci95': 0, 'n': 0}
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {'mean': 0, 'std': 0, 'ci95': 0, 'n': 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {'mean': mean, 'std': std, 'ci95': ci95, 'n': len(arr)}


def fmt_ci(m):
    if m['n'] == 0:
        return "—"
    return f"{m['mean']:.3f} ± {m['ci95']:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted', required=True, type=str)
    parser.add_argument('--processed', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
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
    with open(output_dir / 'ablation_v2_config.json', 'w', encoding='utf-8') as f:
        json.dump({**config, 'seeds': args.seeds, 'input_window': args.input_window},
                  f, ensure_ascii=False, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  4-Hyperedge Ablation v2 (5-fold × 5-seed)")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}, {p.total_memory/1e9:.1f} GB")
    print(f"  Seeds: {args.seeds}")
    print(f"  Conditions: 5 (full B6 + 4 ablations)")
    print(f"  Scenarios:  early + full")
    print(f"  Total runs: 5 conditions × 5 folds × {len(args.seeds)} seeds × 2 scenarios = "
          f"{5 * 5 * len(args.seeds) * 2}\n")

    by_patient, patient_labels, fold_tr = load_data(args)

    runs_path = output_dir / 'ablation_v2_runs.jsonl'
    runs_file = open(runs_path, 'w', encoding='utf-8')

    total_runs = 5 * 5 * len(args.seeds) * 2
    run_idx = 0
    overall_start = time.time()

    conditions = [
        (None, 'full_B6'),
        (0, 'minus E_trajectory'),
        (1, 'minus E_co-trajectory'),
        (2, 'minus E_intervention-response'),
        (3, 'minus E_clinical_prior'),
    ]

    for scenario in ['early', 'full']:
        print(f"\n{'─'*70}")
        print(f"  Scenario: {scenario}")
        print(f"{'─'*70}")
        print(f"  그래프 빌드 ...")
        graphs = build_graphs(
            by_patient, patient_labels, args.embeddings, scenario, args.input_window,
        )
        n_tr_valid = sum(1 for g in graphs.values() if g.tr_valid)
        print(f"  → {len(graphs):,} 환자 / TR-valid {n_tr_valid}명")

        for fold_info in fold_tr.folds:
            fold = fold_info.fold_index
            train_pids = set(fold_info.train_patients)
            test_pids = set(fold_info.test_patients)

            train_graphs_all = sorted(
                [g for pid, g in graphs.items() if pid in train_pids],
                key=lambda g: g.patient_id,
            )
            test_graphs = sorted(
                [g for pid, g in graphs.items() if pid in test_pids],
                key=lambda g: g.patient_id,
            )
            n_val = max(1, int(len(train_graphs_all) * 0.1))
            val_graphs = train_graphs_all[:n_val]
            train_graphs_actual = train_graphs_all[n_val:]

            from training.graph_trainer import compute_diagnosis_class_weights
            diag_weights = compute_diagnosis_class_weights(train_graphs_actual)

            print(f"\n  Fold {fold} (train {len(train_graphs_actual)}, val {len(val_graphs)}, "
                  f"test {len(test_graphs)}):")

            for seed in args.seeds:
                for excluded_type, cond_name in conditions:
                    run_idx += 1
                    start = time.time()
                    result = run_single(
                        train_graphs_actual, val_graphs, test_graphs,
                        diag_weights, config, device,
                        scenario, fold, seed, excluded_type,
                    )
                    elapsed = time.time() - start
                    runs_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    runs_file.flush()

                    total_elapsed = time.time() - overall_start
                    avg = total_elapsed / run_idx
                    remaining = (total_runs - run_idx) * avg
                    print(f"    [{run_idx:>3}/{total_runs}] f={fold} s={seed} {cond_name:<32s} "
                          f"AUROC={result['test_tr']['auroc']:.3f} "
                          f"diag_f1={result['test_diag']['macro_f1']:.3f} "
                          f"({elapsed:.1f}s, ~{remaining/60:.1f}분 남음)")

        del graphs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    runs_file.close()
    overall_time = time.time() - overall_start

    # ===== Aggregate =====
    print(f"\n{'='*70}")
    print(f"  집계 (총 {overall_time/60:.1f}분)")
    print(f"{'='*70}\n")

    all_runs = []
    with open(runs_path) as f:
        for line in f:
            all_runs.append(json.loads(line))

    # Group by (condition, scenario)
    grouped = defaultdict(list)
    for r in all_runs:
        grouped[(r['condition'], r['scenario'])].append(r)

    aggregated = {}
    for (cond, scen), runs in grouped.items():
        tr_aurocs = [r['test_tr']['auroc'] for r in runs]
        tr_f1s = [r['test_tr']['f1_pos'] for r in runs]
        tr_mccs = [r['test_tr']['mcc'] for r in runs]
        tr_baccs = [r['test_tr']['balanced_acc'] for r in runs]
        diag_f1s = [r['test_diag']['macro_f1'] for r in runs]
        diag_accs = [r['test_diag']['accuracy'] for r in runs]

        # Per-diagnosis TR aurocs (only valid ones)
        per_diag_aurocs = defaultdict(list)
        for r in runs:
            for dn, dm in r.get('test_per_diag_tr', {}).items():
                if dm is not None and dm.get('auroc') is not None:
                    per_diag_aurocs[dn].append(dm['auroc'])

        aggregated[f"{cond}_{scen}"] = {
            'condition': cond, 'scenario': scen, 'n_runs': len(runs),
            'tr_auroc': aggregate_metrics_with_ci(tr_aurocs),
            'tr_f1_pos': aggregate_metrics_with_ci(tr_f1s),
            'tr_mcc': aggregate_metrics_with_ci(tr_mccs),
            'tr_balanced_acc': aggregate_metrics_with_ci(tr_baccs),
            'diag_macro_f1': aggregate_metrics_with_ci(diag_f1s),
            'diag_accuracy': aggregate_metrics_with_ci(diag_accs),
            'per_diag_auroc': {dn: aggregate_metrics_with_ci(vals)
                               for dn, vals in per_diag_aurocs.items()},
        }

    with open(output_dir / 'ablation_v2_aggregated.json', 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"  ✓ ablation_v2_aggregated.json saved")

    # ===== Markdown summary =====
    md = [f"# 4-Hyperedge Ablation 결과 (5-fold × {len(args.seeds)}-seed, 각 condition n={5*len(args.seeds)} runs)\n"]
    md.append(f"\n총 학습 시간: {overall_time/60:.1f}분 ({len(all_runs)} runs)\n")

    cond_order = ['full_B6', 'minus_E_trajectory', 'minus_E_co-trajectory',
                  'minus_E_intervention-response', 'minus_E_clinical_prior']
    cond_display = {
        'full_B6': '**full B6 (baseline)**',
        'minus_E_trajectory': '− E_trajectory',
        'minus_E_co-trajectory': '− E_co-trajectory',
        'minus_E_intervention-response': '− E_intervention-response',
        'minus_E_clinical_prior': '− E_clinical_prior',
    }

    for scenario in ['early', 'full']:
        md.append(f"\n## Scenario: {scenario}\n")
        md.append(f"\n### TR (binary classification)\n")
        md.append("| Condition | AUROC | F1_pos | MCC | Bal Acc | Δ AUROC vs baseline |")
        md.append("|---|---|---|---|---|---|")

        baseline_key = f"full_B6_{scenario}"
        baseline_auroc = aggregated[baseline_key]['tr_auroc']['mean']

        for cond in cond_order:
            key = f"{cond}_{scenario}"
            a = aggregated[key]
            delta = a['tr_auroc']['mean'] - baseline_auroc
            delta_str = "—" if cond == 'full_B6' else f"{delta:+.3f}"
            md.append(f"| {cond_display[cond]} | {fmt_ci(a['tr_auroc'])} | "
                      f"{fmt_ci(a['tr_f1_pos'])} | {fmt_ci(a['tr_mcc'])} | "
                      f"{fmt_ci(a['tr_balanced_acc'])} | {delta_str} |")

        md.append(f"\n### Diagnosis (4-class)\n")
        md.append("| Condition | Macro F1 | Accuracy | Δ F1 vs baseline |")
        md.append("|---|---|---|---|")
        baseline_diag = aggregated[baseline_key]['diag_macro_f1']['mean']
        for cond in cond_order:
            key = f"{cond}_{scenario}"
            a = aggregated[key]
            delta = a['diag_macro_f1']['mean'] - baseline_diag
            delta_str = "—" if cond == 'full_B6' else f"{delta:+.3f}"
            md.append(f"| {cond_display[cond]} | {fmt_ci(a['diag_macro_f1'])} | "
                      f"{fmt_ci(a['diag_accuracy'])} | {delta_str} |")

        md.append(f"\n### 진단군별 TR AUROC\n")
        diags = ['depression', 'anxiety', 'addiction']
        md.append("| Condition | " + " | ".join([d.capitalize() for d in diags]) + " |")
        md.append("|---|" + "---|" * len(diags))
        for cond in cond_order:
            key = f"{cond}_{scenario}"
            a = aggregated[key]
            row = [cond_display[cond]]
            for d in diags:
                pd = a['per_diag_auroc'].get(d, {'n': 0})
                row.append(fmt_ci(pd))
            md.append("| " + " | ".join(row) + " |")

    md_str = '\n'.join(md)
    with open(output_dir / 'ablation_v2_summary.md', 'w', encoding='utf-8') as f:
        f.write(md_str)
    print(f"  ✓ ablation_v2_summary.md saved\n")
    print(md_str)

    print(f"\n{'='*70}")
    print(f"  ✓ 4-Hyperedge Ablation v2 완료 ({len(all_runs)} runs, {overall_time/60:.1f}분)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
