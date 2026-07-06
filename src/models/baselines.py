"""
C-block 1차 (v3): B1/B2/B3 baseline 모델.

[v3 변경사항 vs v2]
- 입력 차원이 임상 심각도 features (+6 dims)에 맞춰 변경:
    B1: 1024 → 1030 (+ severity_first_3, severity_mean_input)
    B2: 1034 → 1040
    B3: 1090 → 1096

다른 구조 변경:
- LayerNorm input은 유지
- hidden_dim=128, dropout=0.4 유지 (v2와 동일)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskMLP(nn.Module):
    """환자 feature → MLP → (TR logit, 진단 4-class logits)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_diagnosis: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_diagnosis = num_diagnosis

        self.input_norm = nn.LayerNorm(input_dim)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tr_head = nn.Linear(hidden_dim // 2, 1)
        self.diag_head = nn.Linear(hidden_dim // 2, num_diagnosis)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.input_norm(x)
        h = self.backbone(x)
        tr_logit = self.tr_head(h).squeeze(-1)
        diag_logits = self.diag_head(h)
        return {
            'tr_logit': tr_logit,
            'diag_logits': diag_logits,
            'hidden': h,
        }


# ============================================================================
# Baselines B1, B2, B3 (v3 dimensions)
# ============================================================================
class BaselineB1(MultiTaskMLP):
    """B1 (v3): mean_emb + delta_emb + severity_first_3 + severity_mean_input."""
    def __init__(self, **kwargs):
        super().__init__(input_dim=1030, **kwargs)


class BaselineB2(MultiTaskMLP):
    """B2 (v3): B1 + mean_meta + delta_meta."""
    def __init__(self, **kwargs):
        super().__init__(input_dim=1040, **kwargs)


class BaselineB3(MultiTaskMLP):
    """B3 (v3): B2 + mean_symptom_28 + delta_symptom_28."""
    def __init__(self, **kwargs):
        super().__init__(input_dim=1096, **kwargs)


# ============================================================================
# Multi-task loss (변경 없음)
# ============================================================================
def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask].float()

    bce = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction='none')
    p = torch.sigmoid(valid_logits)
    pt = torch.where(valid_targets > 0.5, p, 1 - p)
    focal_weight = (1 - pt) ** gamma
    alpha_t = torch.where(
        valid_targets > 0.5,
        torch.full_like(valid_targets, alpha),
        torch.full_like(valid_targets, 1 - alpha),
    )
    loss = alpha_t * focal_weight * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def diagnosis_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=class_weights)


def multitask_loss(
    outputs: dict,
    batch: dict,
    lambda_tr: float = 1.0,
    lambda_diag: float = 0.3,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    diag_class_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    tr_logit = outputs['tr_logit']
    diag_logits = outputs['diag_logits']

    tr_target = batch['tr_label']
    tr_valid = batch['tr_valid'].bool()
    diag_target = batch['diag_label']

    loss_tr = focal_binary_loss(
        tr_logit, tr_target, tr_valid,
        alpha=focal_alpha, gamma=focal_gamma,
    )
    loss_diag = diagnosis_loss(diag_logits, diag_target, diag_class_weights)
    total = lambda_tr * loss_tr + lambda_diag * loss_diag

    return total, {
        'total': total.item(),
        'loss_tr': loss_tr.item() if isinstance(loss_tr, torch.Tensor) else 0.0,
        'loss_diag': loss_diag.item(),
        'n_tr_valid': int(tr_valid.sum().item()),
    }


MODEL_REGISTRY = {
    'B1': BaselineB1,
    'B2': BaselineB2,
    'B3': BaselineB3,
}


def build_model(model_name: str, **kwargs) -> MultiTaskMLP:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)


def get_feature_key(model_name: str) -> str:
    return {'B1': 'b1_feature', 'B2': 'b2_feature', 'B3': 'b3_feature'}[model_name]
