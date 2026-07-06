"""
C-block 2차: 그래프 모델 학습 루프.

각 환자별 그래프 크기가 다르므로 padding + masking 방식으로 batching.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Patient graph Dataset
# ============================================================================
class PatientGraphDataset(Dataset):
    """PatientGraph 리스트를 Dataset으로."""

    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def make_collate_fn(model_type: str):
    """모델 종류에 따른 collate function 생성."""
    from graph.build_graph import pad_and_batch_graphs

    def _collate(batch):
        return pad_and_batch_graphs(batch, model_type=model_type)

    return _collate


# ============================================================================
# 학습 결과 데이터 클래스
# ============================================================================
@dataclass
class TrainResult:
    model_name: str
    scenario: str
    fold: int
    seed: int
    epochs_trained: int
    best_epoch: int
    best_val_auroc: float
    best_val_metrics: Dict
    test_metrics: Dict
    test_diag_metrics: Dict
    test_per_diag_tr: Dict
    train_time_sec: float


# ============================================================================
# 진단 클래스 가중치
# ============================================================================
def compute_diagnosis_class_weights(graphs, n_classes: int = 4) -> torch.Tensor:
    counter = Counter(g.diag_label for g in graphs)
    total = sum(counter.values())
    weights = torch.zeros(n_classes)
    for c in range(n_classes):
        n = counter.get(c, 1)
        weights[c] = total / (n_classes * n)
    return weights


# ============================================================================
# 평가
# ============================================================================
@torch.no_grad()
def evaluate(model, loader, device, full_metrics: bool = False) -> dict:
    from training.metrics import compute_diag_metrics, compute_per_diagnosis_tr, compute_tr_metrics

    model.eval()
    all_tr_logits = []
    all_tr_labels = []
    all_tr_valid = []
    all_diag_logits = []
    all_diag_labels = []

    for batch in loader:
        for k in ('node_features', 'node_mask', 'severity_first_3', 'severity_mean_input',
                  'tr_label', 'tr_valid', 'diag_label'):
            if k in batch:
                batch[k] = batch[k].to(device)
        for k in ('adjacency', 'incidence', 'edge_mask', 'edge_types'):
            if k in batch:
                batch[k] = batch[k].to(device)

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
        tr_metrics = compute_tr_metrics(tr_labels[tr_valid], tr_probs[tr_valid])
    else:
        tr_metrics = compute_tr_metrics(np.array([]), np.array([]))

    diag_metrics = compute_diag_metrics(diag_labels, diag_preds, diag_probs)

    result = {'tr': tr_metrics, 'diag': diag_metrics}
    if full_metrics:
        result['per_diag_tr'] = compute_per_diagnosis_tr(
            tr_labels, tr_probs, diag_labels, tr_valid,
        )
    return result


# ============================================================================
# 단일 (fold, seed) 학습
# ============================================================================
def train_single_run(
    model_name: str,
    train_graphs: list,
    val_graphs: list,
    test_graphs: list,
    diagnosis_class_weights: Optional[torch.Tensor],
    config: dict,
    device: torch.device,
    scenario: str = 'early',
    fold: int = 0,
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    from models.graph_models import build_graph_model, multitask_loss

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

    if diagnosis_class_weights is not None:
        diagnosis_class_weights = diagnosis_class_weights.to(device)

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
                diag_class_weights=diagnosis_class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_loss_avg = train_loss_sum / max(n_batches, 1)

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            current_auroc = val_metrics['tr']['auroc']
        else:
            val_metrics = None
            current_auroc = 0.0

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"  [seed {seed} fold {fold} ep {epoch+1}] train_loss {train_loss_avg:.4f}"
            if val_metrics is not None:
                msg += f"  val_auroc {current_auroc:.4f}"
            print(msg)

        if val_loader is not None and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  [seed {seed} fold {fold}] early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics_dict = evaluate(model, test_loader, device, full_metrics=True)
    train_time = time.time() - start

    return TrainResult(
        model_name=model_name, scenario=scenario,
        fold=fold, seed=seed,
        epochs_trained=epoch + 1, best_epoch=best_epoch,
        best_val_auroc=best_val_auroc,
        best_val_metrics=val_metrics if val_metrics is not None else {},
        test_metrics=test_metrics_dict['tr'],
        test_diag_metrics=test_metrics_dict['diag'],
        test_per_diag_tr=test_metrics_dict['per_diag_tr'],
        train_time_sec=train_time,
    )
