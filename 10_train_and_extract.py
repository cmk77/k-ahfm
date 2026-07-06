#!/usr/bin/env python
"""
10: §4.8 attention 시각화용 — B6 1-fold 학습 + 체크포인트 저장 + 환자별 attention 추출.

기본 설정 (CLI로 변경 가능):
    - fold 0, seed 42, scenario=early (test_TR AUROC 0.816 — 50 runs 중 1위)
    - B6 (4-edge K-AHFM-Clinic)

산출물 (output_dir 아래):
    - B6_fold0_seed42_early_best.pt        : best 모델 체크포인트
    - train_log.json                        : epoch별 학습 로그
    - patient_predictions.json              : test set 환자별 (TR/diag prediction + true label)
    - patient_attention.npz                 : 환자별 alpha_e1, alpha_e2, beta_v, edge_types, masks
    - train_summary.json                    : 최종 metrics 요약

사용법:
    python 10_train_and_extract.py \\
        --extracted <repo>/data/extracted \\
        --processed <repo>/data/processed \\
        --embeddings <repo>/data/embeddings \\
        --output <repo>/results/b6_attention_fold0_seed42_early
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


def load_data(args):
    from data.parser import parse_all_sessions
    from data.labels import load_labels_json
    from data.splits import SplitResult

    print(f"[1/3] Sessions 파싱: {args.extracted}/training/labeling ...")
    labeling = Path(args.extracted) / 'training' / 'labeling'
    sessions = parse_all_sessions(labeling, progress=True)
    by_patient = defaultdict(list)
    for s in sessions:
        if s.patient_id and s.session_number > 0:
            by_patient[s.patient_id].append(s)
    for pid in by_patient:
        by_patient[pid].sort(key=lambda s: s.session_number)
    print(f"  → {len(sessions):,} 세션 / {len(by_patient):,} 환자")

    labels_path = Path(args.processed) / 'patient_labels.json'
    patient_labels = load_labels_json(labels_path)
    print(f"[2/3] 환자 라벨 로드: {len(patient_labels):,}명")

    fold_tr = SplitResult.from_json(Path(args.processed) / 'fold_tr.json')
    print(f"[3/3] Fold TR: {fold_tr.n_patients}명, {fold_tr.n_folds}-fold")

    return dict(by_patient), patient_labels, fold_tr


def train_with_checkpoint(
    model_name: str,
    train_graphs: list,
    val_graphs: list,
    test_graphs: list,
    diag_weights: torch.Tensor,
    config: dict,
    device: torch.device,
    scenario: str,
    fold: int,
    seed: int,
    output_dir: Path,
    verbose: bool = True,
):
    """train_single_run의 변종 — best state를 디스크에 저장 + 학습 로그 반환."""
    from models.graph_models import build_graph_model, multitask_loss
    from training.graph_trainer import (
        PatientGraphDataset, make_collate_fn, evaluate,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = build_graph_model(
        model_name, hidden_dim=config['hidden_dim'], dropout=config['dropout'],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
    )

    collate = make_collate_fn(model_name)
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

    best_val_auroc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience = config['patience']
    log = []

    start = time.time()
    for epoch in range(config['max_epochs']):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            for k in ('node_features', 'node_mask', 'severity_first_3', 'severity_mean_input',
                      'tr_label', 'tr_valid', 'diag_label'):
                if k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
            for k in ('adjacency', 'incidence', 'edge_mask', 'edge_types'):
                if k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

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

        train_loss_avg = train_loss_sum / max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else None
        current_auroc = val_metrics['tr']['auroc'] if val_metrics else 0.0

        log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'val_tr_auroc': current_auroc,
            'val_tr_f1': val_metrics['tr']['f1_pos'] if val_metrics else 0.0,
            'val_diag_f1': val_metrics['diag']['macro_f1'] if val_metrics else 0.0,
        })

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  ep {epoch+1:3d}  train_loss {train_loss_avg:.4f}  "
                  f"val_auroc {current_auroc:.4f}  val_diag_f1 {log[-1]['val_diag_f1']:.4f}")

        if val_loader is not None and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  [Early stop] epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = output_dir / f'{model_name}_fold{fold}_seed{seed}_{scenario}_best.pt'
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'scenario': scenario,
        'fold': fold,
        'seed': seed,
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch,
        'epochs_trained': epoch + 1,
    }, ckpt_path)
    print(f"\n  ✓ Checkpoint saved: {ckpt_path}")

    # Test evaluation
    test_metrics = evaluate(model, test_loader, device, full_metrics=True)
    train_time = time.time() - start

    return {
        'model': model,
        'test_metrics': test_metrics,
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch,
        'epochs_trained': epoch + 1,
        'train_time_sec': train_time,
        'log': log,
        'test_graphs': test_graphs,
        'val_graphs': val_graphs,
    }


@torch.no_grad()
def extract_attention_per_patient(model, graphs, device, model_type='B6', desc=''):
    """env 환자별로 inference (batch_size=8) + attention 추출."""
    from training.graph_trainer import PatientGraphDataset, make_collate_fn

    model.eval()
    ds = PatientGraphDataset(graphs)
    collate = make_collate_fn(model_type)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, drop_last=False)

    results = []
    for batch_idx, batch in enumerate(loader):
        for k in ('node_features', 'node_mask', 'severity_first_3', 'severity_mean_input',
                  'tr_label', 'tr_valid', 'diag_label'):
            if k in batch:
                batch[k] = batch[k].to(device)
        for k in ('adjacency', 'incidence', 'edge_mask', 'edge_types'):
            if k in batch:
                batch[k] = batch[k].to(device)

        outputs = model(batch, return_attention=True)

        bs = batch['node_features'].size(0)
        start = batch_idx * 8
        for i in range(bs):
            g = graphs[start + i]
            tr_logit = outputs['tr_logit'][i].item()
            tr_prob = float(torch.sigmoid(torch.tensor(tr_logit)).item())
            diag_logits = outputs['diag_logits'][i].cpu().numpy()
            diag_prob = float(torch.softmax(torch.tensor(diag_logits), dim=-1).max().item())
            diag_pred = int(diag_logits.argmax())

            results.append({
                'patient_id': g.patient_id,
                'diagnosis': g.diagnosis,
                'num_nodes': int(g.num_nodes),
                'true_tr_label': int(g.tr_label) if hasattr(g, 'tr_label') and g.tr_label is not None else None,
                'tr_valid': bool(g.tr_valid) if hasattr(g, 'tr_valid') else False,
                'true_diag_label': int(g.diag_label) if hasattr(g, 'diag_label') else None,
                'pred_tr_logit': tr_logit,
                'pred_tr_prob': tr_prob,
                'pred_tr_label': int(tr_prob > 0.5),
                'pred_diag_label': diag_pred,
                'pred_diag_prob': diag_prob,
                # attention/hyperedge weights (numpy로 변환, cpu)
                'alpha_e1': outputs['alpha_e1'][i].cpu().numpy(),
                'alpha_e2': outputs['alpha_e2'][i].cpu().numpy(),
                'beta_v': outputs['beta_v'][i].cpu().numpy(),
                'edge_types': outputs['edge_types'][i].cpu().numpy(),
                'node_mask': outputs['node_mask'][i].cpu().numpy(),
                'edge_mask': outputs['edge_mask'][i].cpu().numpy(),
            })
    if desc:
        print(f"  ✓ {desc}: {len(results)} patients")
    return results


def save_attention_outputs(results, output_dir, split_name='test'):
    """attention + predictions 저장."""
    # JSON-serializable predictions
    predictions = []
    for r in results:
        predictions.append({
            'patient_id': r['patient_id'],
            'diagnosis': r['diagnosis'],
            'num_nodes': r['num_nodes'],
            'true_tr_label': r['true_tr_label'],
            'tr_valid': r['tr_valid'],
            'true_diag_label': r['true_diag_label'],
            'pred_tr_logit': r['pred_tr_logit'],
            'pred_tr_prob': r['pred_tr_prob'],
            'pred_tr_label': r['pred_tr_label'],
            'pred_diag_label': r['pred_diag_label'],
            'pred_diag_prob': r['pred_diag_prob'],
        })

    pred_path = output_dir / f'patient_predictions_{split_name}.json'
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Predictions saved: {pred_path}")

    # Attention as .npz (numpy archive)
    # Each patient gets prefixed keys
    npz_dict = {}
    patient_ids = []
    for r in results:
        pid = r['patient_id']
        patient_ids.append(pid)
        npz_dict[f'{pid}__alpha_e1'] = r['alpha_e1']
        npz_dict[f'{pid}__alpha_e2'] = r['alpha_e2']
        npz_dict[f'{pid}__beta_v'] = r['beta_v']
        npz_dict[f'{pid}__edge_types'] = r['edge_types']
        npz_dict[f'{pid}__node_mask'] = r['node_mask']
        npz_dict[f'{pid}__edge_mask'] = r['edge_mask']
    npz_dict['__patient_ids'] = np.array(patient_ids)

    attn_path = output_dir / f'patient_attention_{split_name}.npz'
    np.savez_compressed(attn_path, **npz_dict)
    print(f"  ✓ Attention saved: {attn_path}")


def main():
    parser = argparse.ArgumentParser(description="B6 1-fold 학습 + attention 추출")
    parser.add_argument('--extracted', required=True, type=str)
    parser.add_argument('--processed', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--model', default='B6', choices=['B6', 'B6v2'])
    parser.add_argument('--scenario', default='early', choices=['early', 'full'])
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

    parser.add_argument('--emotion-features', type=str, default=None,
                        help='B6v2 학습 시만 필수')
    args = parser.parse_args()

    if args.model == 'B6v2' and args.emotion_features is None:
        sys.exit("ERROR: B6v2는 --emotion-features 필수")

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
    with open(output_dir / 'train_config.json', 'w', encoding='utf-8') as f:
        json.dump({**config, 'model': args.model, 'scenario': args.scenario,
                   'fold': args.fold, 'seed': args.seed,
                   'input_window': args.input_window}, f, ensure_ascii=False, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  B6 1-fold 학습 + attention 추출 (§4.8용)")
    print(f"{'='*70}")
    print(f"  Device:   {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU:      {p.name}, {p.total_memory / 1e9:.1f} GB")
    print(f"  Model:    {args.model}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Fold/Seed: {args.fold} / {args.seed}")
    print(f"  Output:   {output_dir}")

    by_patient, patient_labels, fold_tr = load_data(args)

    # Build graphs
    from graph.build_graph import build_all_patient_graphs
    print(f"\n[그래프 빌드: {args.model} / {args.scenario}]")
    graphs = build_all_patient_graphs(
        sessions_by_patient=by_patient,
        patient_labels=patient_labels,
        embeddings_cache_dir=Path(args.embeddings),
        model_type=args.model,
        scenario=args.scenario,
        input_window=args.input_window,
        progress=True,
        emotion_features_path=Path(args.emotion_features) if args.emotion_features else None,
    )
    n_tr_valid = sum(1 for g in graphs.values() if g.tr_valid)
    print(f"  → {len(graphs):,} 환자 / TR-valid {n_tr_valid}명")

    # Fold split
    fold_info = next(f for f in fold_tr.folds if f.fold_index == args.fold)
    train_pids = set(fold_info.train_patients)
    test_pids = set(fold_info.test_patients)

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

    # Diag class weights
    from training.graph_trainer import compute_diagnosis_class_weights
    diag_weights = compute_diagnosis_class_weights(train_graphs_actual)

    # Train
    print(f"\n[학습 시작]")
    result = train_with_checkpoint(
        model_name=args.model,
        train_graphs=train_graphs_actual,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        diag_weights=diag_weights,
        config=config,
        device=device,
        scenario=args.scenario,
        fold=args.fold,
        seed=args.seed,
        output_dir=output_dir,
    )

    # Save train log
    with open(output_dir / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump(result['log'], f, ensure_ascii=False, indent=2)

    # Save train summary
    summary = {
        'model': args.model,
        'scenario': args.scenario,
        'fold': args.fold,
        'seed': args.seed,
        'best_val_auroc': result['best_val_auroc'],
        'best_epoch': result['best_epoch'],
        'epochs_trained': result['epochs_trained'],
        'train_time_sec': result['train_time_sec'],
        'test_tr_metrics': result['test_metrics']['tr'],
        'test_diag_metrics': result['test_metrics']['diag'],
        'test_per_diag_tr': result['test_metrics']['per_diag_tr'],
    }
    with open(output_dir / 'train_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ train_summary.json saved")
    print(f"     best_val_auroc: {result['best_val_auroc']:.4f}")
    print(f"     test_tr_auroc:  {result['test_metrics']['tr']['auroc']:.4f}")
    print(f"     test_diag_f1:   {result['test_metrics']['diag']['macro_f1']:.4f}")
    print(f"     train_time:     {result['train_time_sec']:.1f}s")

    # Extract attention for ALL splits (train/val/test)
    print(f"\n[Attention 추출]")
    model = result['model']
    test_results = extract_attention_per_patient(model, test_graphs, device, args.model, desc='test')
    save_attention_outputs(test_results, output_dir, 'test')

    val_results = extract_attention_per_patient(model, val_graphs, device, args.model, desc='val')
    save_attention_outputs(val_results, output_dir, 'val')

    # train도 추출 (case study에서 train 환자도 사용 가능)
    train_results = extract_attention_per_patient(model, train_graphs_actual, device, args.model, desc='train')
    save_attention_outputs(train_results, output_dir, 'train')

    print(f"\n{'='*70}")
    print(f"  ✓ 학습 + Attention 추출 완료")
    print(f"  산출물: {output_dir}/")
    print(f"    - {args.model}_fold{args.fold}_seed{args.seed}_{args.scenario}_best.pt")
    print(f"    - train_log.json, train_summary.json")
    print(f"    - patient_predictions_{{train,val,test}}.json")
    print(f"    - patient_attention_{{train,val,test}}.npz")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
