"""
C-block 2차: 평가 메트릭 (cblock1과 동일).

TR 이진: AUROC, F1(pos), MCC, Balanced Accuracy
진단 4-class: Macro F1, Accuracy, per-class P/R
진단군별 TR 분해: depression/anxiety/addiction별 AUROC/F1/MCC
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score,
)


def compute_tr_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """TR 이진 분류 메트릭."""
    if len(y_true) == 0:
        return {'auroc': 0.0, 'f1_pos': 0.0, 'mcc': 0.0,
                'balanced_acc': 0.0, 'accuracy': 0.0,
                'precision_pos': 0.0, 'recall_pos': 0.0,
                'n_samples': 0, 'n_pos': 0, 'n_neg': 0}

    y_pred = (y_prob >= threshold).astype(np.int32)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    metrics = {
        'n_samples': len(y_true),
        'n_pos': n_pos,
        'n_neg': n_neg,
    }

    if n_pos > 0 and n_neg > 0:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['auroc'] = 0.0
    else:
        metrics['auroc'] = 0.0

    try:
        metrics['f1_pos'] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    except Exception:
        metrics['f1_pos'] = 0.0

    try:
        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics['mcc'] = 0.0

    try:
        metrics['balanced_acc'] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        metrics['balanced_acc'] = 0.0

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0,
    )
    metrics['precision_pos'] = float(p[0]) if len(p) > 0 else 0.0
    metrics['recall_pos'] = float(r[0]) if len(r) > 0 else 0.0

    return metrics


DIAGNOSIS_NAMES = ['depression', 'anxiety', 'addiction', 'normal']


def compute_diag_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, list]]:
    """4-class 진단 메트릭."""
    if len(y_true) == 0:
        return {'macro_f1': 0.0, 'accuracy': 0.0,
                'per_class_f1': [0.0]*4,
                'per_class_precision': [0.0]*4,
                'per_class_recall': [0.0]*4,
                'n_samples': 0}

    metrics = {'n_samples': int(len(y_true))}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0,
    )
    metrics['per_class_precision'] = [float(x) for x in p]
    metrics['per_class_recall'] = [float(x) for x in r]
    metrics['per_class_f1'] = [float(x) for x in f]
    metrics['per_class_support'] = [int(x) for x in sup]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def compute_per_diagnosis_tr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    diag_labels: np.ndarray,
    tr_valid_mask: np.ndarray,
) -> Dict[str, Optional[Dict[str, float]]]:
    """진단군별 TR 분해 (표 4.2용)."""
    result: Dict[str, Optional[Dict[str, float]]] = {}
    for c, name in enumerate(DIAGNOSIS_NAMES):
        if name == 'normal':
            result[name] = None
            continue
        mask = (diag_labels == c) & tr_valid_mask
        if mask.sum() == 0:
            result[name] = None
            continue
        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        result[name] = compute_tr_metrics(sub_y, sub_p)
    return result


def aggregate_metrics(
    metric_runs: List[Dict[str, float]],
    keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """반복 학습 결과 평균 ± 95% CI."""
    if not metric_runs:
        return {}

    if keys is None:
        keys = [k for k in metric_runs[0].keys()
                if isinstance(metric_runs[0][k], (int, float))]

    agg = {}
    for k in keys:
        values = [r[k] for r in metric_runs if k in r and isinstance(r[k], (int, float))]
        if not values:
            agg[k] = {'mean': 0.0, 'std': 0.0, 'ci95': 0.0, 'n': 0}
            continue
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        agg[k] = {
            'mean': mean,
            'std': std,
            'ci95': float(ci95),
            'n': len(arr),
        }
    return agg


def format_mean_ci(agg_metric: Dict[str, float], digits: int = 3) -> str:
    if not agg_metric or 'mean' not in agg_metric:
        return "—"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(agg_metric['mean'])} ± {fmt.format(agg_metric['ci95'])}"
