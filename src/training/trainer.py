"""
C-block 1차: B1/B2/B3 baseline 학습 루프.

핵심 설계:
    - 환자 단위 학습 (각 batch는 환자들의 collection)
    - Multi-task loss: focal(TR) + weighted CE(진단)
    - Early stopping: val AUROC 기준 (patience=15)
    - 각 (fold, seed) 조합당 별도 학습
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# 환자 feature → PyTorch Dataset
# ============================================================================
class PatientFeatureDataset(Dataset):
    """환자 feature 리스트를 PyTorch Dataset으로."""

    def __init__(self, features: list, feature_key: str):
        """
        Args:
            features: PatientFeature 리스트
            feature_key: 'b1_feature' | 'b2_feature' | 'b3_feature'
        """
        self.features = features
        self.feature_key = feature_key

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict:
        f = self.features[idx]
        return {
            'feature': torch.from_numpy(getattr(f, self.feature_key)).float(),
            'tr_label': torch.tensor(max(f.tr_label, 0), dtype=torch.long),
            'tr_valid': torch.tensor(1 if f.tr_valid else 0, dtype=torch.long),
            'diag_label': torch.tensor(f.diag_label, dtype=torch.long),
            'patient_id': f.patient_id,
            'diagnosis': f.diagnosis,
        }


def collate_patient_features(batch):
    """환자 feature batch."""
    return {
        'feature': torch.stack([b['feature'] for b in batch]),
        'tr_label': torch.stack([b['tr_label'] for b in batch]),
        'tr_valid': torch.stack([b['tr_valid'] for b in batch]),
        'diag_label': torch.stack([b['diag_label'] for b in batch]),
        'patient_id': [b['patient_id'] for b in batch],
        'diagnosis': [b['diagnosis'] for b in batch],
    }


# ============================================================================
# 학습 결과
# ============================================================================
@dataclass
class TrainResult:
    """단일 (fold, seed) 학습 결과."""
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
# 학습 루프
# ============================================================================
def train_single_run(
    model_name: str,
    train_features: list,
    val_features: list,
    test_features: list,
    feature_key: str,
    diagnosis_class_weights: Optional[torch.Tensor],
    config: dict,
    device: torch.device,
    scenario: str = 'early',
    fold: int = 0,
    seed: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """단일 (fold, seed) 학습 → 평가."""
    from models.baselines import build_model, multitask_loss
    from training.metrics import compute_diag_metrics, compute_per_diagnosis_tr, compute_tr_metrics

    # 시드
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # 모델
    model = build_model(model_name, hidden_dim=config['hidden_dim'],
                        dropout=config['dropout']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # DataLoaders
    train_ds = PatientFeatureDataset(train_features, feature_key)
    val_ds = PatientFeatureDataset(val_features, feature_key) if val_features else None
    test_ds = PatientFeatureDataset(test_features, feature_key)

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_patient_features, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'] * 2, shuffle=False,
        collate_fn=collate_patient_features,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=config['batch_size'] * 2, shuffle=False,
        collate_fn=collate_patient_features,
    )

    # Class weights → device
    if diagnosis_class_weights is not None:
        diagnosis_class_weights = diagnosis_class_weights.to(device)

    # Early stopping
    best_val_auroc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience = config['patience']

    start = time.time()
    for epoch in range(config['max_epochs']):
        # ===== Train =====
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch['feature'].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss, _ = multitask_loss(
                outputs,
                {
                    'tr_label': batch['tr_label'].to(device),
                    'tr_valid': batch['tr_valid'].to(device),
                    'diag_label': batch['diag_label'].to(device),
                },
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

        # ===== Validation =====
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            current_auroc = val_metrics['tr']['auroc']
        else:
            # val 없으면 train 자체로 평가 (early stopping 미사용)
            val_metrics = None
            current_auroc = 0.0

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"  [seed {seed} fold {fold} ep {epoch+1}] train_loss {train_loss_avg:.4f}"
            if val_metrics is not None:
                msg += f"  val_auroc {current_auroc:.4f}  val_f1 {val_metrics['tr']['f1_pos']:.4f}"
            print(msg)

        # Best 체크포인트
        if val_loader is not None and current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"  [seed {seed} fold {fold}] early stop at epoch {epoch+1}")
            break

    # ===== Best 모델 로드 후 Test =====
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics_dict = evaluate(model, test_loader, device, full_metrics=True)
    train_time = time.time() - start

    return TrainResult(
        model_name=model_name,
        scenario=scenario,
        fold=fold,
        seed=seed,
        epochs_trained=epoch + 1,
        best_epoch=best_epoch,
        best_val_auroc=best_val_auroc,
        best_val_metrics=val_metrics if val_metrics is not None else {},
        test_metrics=test_metrics_dict['tr'],
        test_diag_metrics=test_metrics_dict['diag'],
        test_per_diag_tr=test_metrics_dict['per_diag_tr'],
        train_time_sec=train_time,
    )


# ============================================================================
# 평가
# ============================================================================
@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    full_metrics: bool = False,
) -> Dict:
    """모델 평가."""
    from training.metrics import compute_diag_metrics, compute_per_diagnosis_tr, compute_tr_metrics

    model.eval()
    all_tr_logits = []
    all_tr_labels = []
    all_tr_valid = []
    all_diag_logits = []
    all_diag_labels = []

    for batch in loader:
        features = batch['feature'].to(device)
        outputs = model(features)
        all_tr_logits.append(outputs['tr_logit'].cpu())
        all_tr_labels.append(batch['tr_label'])
        all_tr_valid.append(batch['tr_valid'])
        all_diag_logits.append(outputs['diag_logits'].cpu())
        all_diag_labels.append(batch['diag_label'])

    tr_logits = torch.cat(all_tr_logits)
    tr_labels = torch.cat(all_tr_labels).numpy()
    tr_valid = torch.cat(all_tr_valid).numpy().astype(bool)
    diag_logits = torch.cat(all_diag_logits)
    diag_labels = torch.cat(all_diag_labels).numpy()

    tr_probs = torch.sigmoid(tr_logits).numpy()
    diag_probs = torch.softmax(diag_logits, dim=-1).numpy()
    diag_preds = diag_logits.argmax(dim=-1).numpy()

    # TR 메트릭 (TR-valid 샘플만)
    if tr_valid.sum() > 0:
        tr_metrics = compute_tr_metrics(tr_labels[tr_valid], tr_probs[tr_valid])
    else:
        tr_metrics = compute_tr_metrics(np.array([]), np.array([]))

    # 진단 메트릭 (모든 환자)
    diag_metrics = compute_diag_metrics(diag_labels, diag_preds, diag_probs)

    result = {'tr': tr_metrics, 'diag': diag_metrics}

    if full_metrics:
        # 진단군별 TR 분해
        per_diag_tr = compute_per_diagnosis_tr(
            tr_labels, tr_probs, diag_labels, tr_valid,
        )
        result['per_diag_tr'] = per_diag_tr

    return result


# ============================================================================
# 클래스 가중치 계산
# ============================================================================
def compute_diagnosis_class_weights(features: list, n_classes: int = 4) -> torch.Tensor:
    """진단 클래스 inverse frequency 가중치."""
    counter = Counter(f.diag_label for f in features)
    total = sum(counter.values())
    weights = torch.zeros(n_classes)
    for c in range(n_classes):
        n = counter.get(c, 1)
        weights[c] = total / (n_classes * n)
    return weights
