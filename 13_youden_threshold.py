#!/usr/bin/env python
"""
13_youden_threshold.py — Youden's J 기반 임계값 보정 분석.

평가자 핵심 비판 대응:
    "Recall_pos=0.114 (35명 중 4명만 식별). 모델이 사실상 무용.
     4 오류 cases의 pred_prob가 0.43-0.50이라는 사실은 모델이 학습한
     결정 경계가 0.5 근처에서 의미 있는 분리를 만들지 못했다는 negative
     signal. Youden's J로 임계값을 0.45 정도로 낮추면 어떻게 되는지
     본 thesis 안에서 계산해서 제시하는 것이 강력하게 권장됩니다."

산출물:
    youden_analysis.json  : threshold + metrics
    youden_analysis.md    : 표 형식
    roc_curve.png         : ROC + Youden's J point

사용법:
    python 13_youden_threshold.py \\
        --predictions <repo>/results/b6_attention_fold0_seed42_early
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, f1_score,
    precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score,
)


def load_predictions(pred_dir):
    """3 splits (train/val/test) predictions 통합 로드."""
    all_preds = []
    for split in ['train', 'val', 'test']:
        pred_path = Path(pred_dir) / f'patient_predictions_{split}.json'
        if not pred_path.exists():
            print(f"  ⚠ {split} not found")
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        for p in preds:
            p['split'] = split
        all_preds.extend(preds)
        print(f"  + {split}: {len(preds)} patients")

    # Filter TR-valid only
    valid_preds = [p for p in all_preds if p.get('tr_valid')]
    print(f"  Total TR-valid: {len(valid_preds)}")
    return valid_preds


def compute_metrics(y_true, y_prob, threshold):
    """주어진 threshold에서 binary metrics 계산."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'threshold': float(threshold),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision_pos': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall_pos': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'recall_neg': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1_pos': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'youden_j': float(tp / (tp + fn) + tn / (tn + fp) - 1) if (tp + fn) > 0 and (tn + fp) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True,
                        help='10_train_and_extract.py output dir (patient_predictions_*.json)')
    parser.add_argument('--output', default=None,
                        help='Output dir (default: predictions dir)')
    args = parser.parse_args()

    pred_dir = Path(args.predictions).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve() if args.output else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading predictions from {pred_dir}/")
    preds = load_predictions(pred_dir)

    # Overall analysis (140 patients)
    y_true = np.array([p['true_tr_label'] for p in preds])
    y_prob = np.array([p['pred_tr_prob'] for p in preds])
    diags = np.array([p['diagnosis'] for p in preds])
    splits = np.array([p['split'] for p in preds])

    print(f"\n[2/4] Computing ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_j = float(j_scores[optimal_idx])

    print(f"  Overall AUROC (140 patients): {auroc:.4f}")
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  Optimal Youden's J: {optimal_j:.4f}")
    print(f"  → Sensitivity (TPR): {tpr[optimal_idx]:.4f}")
    print(f"  → Specificity (1-FPR): {1 - fpr[optimal_idx]:.4f}")

    print(f"\n[3/4] Computing confusion matrices at different thresholds...")
    # Baseline (threshold=0.5)
    metrics_05 = compute_metrics(y_true, y_prob, 0.5)
    # Youden optimal
    metrics_opt = compute_metrics(y_true, y_prob, optimal_threshold)
    # A few other thresholds to compare
    metrics_045 = compute_metrics(y_true, y_prob, 0.45)
    metrics_048 = compute_metrics(y_true, y_prob, 0.48)
    metrics_049 = compute_metrics(y_true, y_prob, 0.49)

    print(f"\n  Baseline (threshold=0.5):")
    print(f"    TP={metrics_05['TP']}, FN={metrics_05['FN']}, FP={metrics_05['FP']}, TN={metrics_05['TN']}")
    print(f"    Recall_pos={metrics_05['recall_pos']:.3f}, Precision_pos={metrics_05['precision_pos']:.3f}, F1={metrics_05['f1_pos']:.3f}, MCC={metrics_05['mcc']:.3f}")

    print(f"\n  Youden's J optimal (threshold={optimal_threshold:.4f}):")
    print(f"    TP={metrics_opt['TP']}, FN={metrics_opt['FN']}, FP={metrics_opt['FP']}, TN={metrics_opt['TN']}")
    print(f"    Recall_pos={metrics_opt['recall_pos']:.3f}, Precision_pos={metrics_opt['precision_pos']:.3f}, F1={metrics_opt['f1_pos']:.3f}, MCC={metrics_opt['mcc']:.3f}")

    print(f"\n  Threshold=0.49:")
    print(f"    TP={metrics_049['TP']}, FN={metrics_049['FN']}, FP={metrics_049['FP']}, TN={metrics_049['TN']}")
    print(f"    Recall_pos={metrics_049['recall_pos']:.3f}, F1={metrics_049['f1_pos']:.3f}, MCC={metrics_049['mcc']:.3f}")

    print(f"\n  Threshold=0.48:")
    print(f"    TP={metrics_048['TP']}, FN={metrics_048['FN']}, FP={metrics_048['FP']}, TN={metrics_048['TN']}")
    print(f"    Recall_pos={metrics_048['recall_pos']:.3f}, F1={metrics_048['f1_pos']:.3f}, MCC={metrics_048['mcc']:.3f}")

    print(f"\n  Threshold=0.45:")
    print(f"    TP={metrics_045['TP']}, FN={metrics_045['FN']}, FP={metrics_045['FP']}, TN={metrics_045['TN']}")
    print(f"    Recall_pos={metrics_045['recall_pos']:.3f}, F1={metrics_045['f1_pos']:.3f}, MCC={metrics_045['mcc']:.3f}")

    # Per-diagnosis analysis at Youden's threshold
    print(f"\n[4/4] Per-diagnosis analysis at Youden's optimal threshold...")
    per_diag = {}
    for diag in ['depression', 'anxiety', 'addiction']:
        mask = diags == diag
        if mask.sum() == 0:
            continue
        diag_y_true = y_true[mask]
        diag_y_prob = y_prob[mask]
        # Baseline 0.5
        bl_metrics = compute_metrics(diag_y_true, diag_y_prob, 0.5)
        # Youden
        op_metrics = compute_metrics(diag_y_true, diag_y_prob, optimal_threshold)
        per_diag[diag] = {
            'n': int(mask.sum()),
            'n_pos': int(diag_y_true.sum()),
            'baseline_0.5': bl_metrics,
            'youden_optimal': op_metrics,
        }
        print(f"\n  {diag.upper()} (n={per_diag[diag]['n']}, n_pos={per_diag[diag]['n_pos']}):")
        print(f"    Baseline:  TP={bl_metrics['TP']}, FN={bl_metrics['FN']}, "
              f"Recall_pos={bl_metrics['recall_pos']:.3f}")
        print(f"    Youden:    TP={op_metrics['TP']}, FN={op_metrics['FN']}, "
              f"Recall_pos={op_metrics['recall_pos']:.3f}")

    # Save outputs
    summary = {
        'n_patients': int(len(y_true)),
        'n_pos': int(y_true.sum()),
        'n_neg': int((1 - y_true).sum()),
        'overall_auroc': float(auroc),
        'pred_prob_distribution': {
            'min': float(y_prob.min()), 'max': float(y_prob.max()),
            'mean': float(y_prob.mean()), 'std': float(y_prob.std()),
            'percentile_05': float(np.percentile(y_prob, 5)),
            'percentile_50': float(np.percentile(y_prob, 50)),
            'percentile_95': float(np.percentile(y_prob, 95)),
        },
        'optimal_threshold': optimal_threshold,
        'optimal_youden_j': optimal_j,
        'metrics_at_0.5': metrics_05,
        'metrics_at_youden_optimal': metrics_opt,
        'metrics_at_0.49': metrics_049,
        'metrics_at_0.48': metrics_048,
        'metrics_at_0.45': metrics_045,
        'per_diagnosis': per_diag,
    }

    with open(out_dir / 'youden_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Saved: {out_dir}/youden_analysis.json")

    # Markdown summary
    md = [
        f"# Youden's J 임계값 보정 분석 (B6 fold 0 / seed 42 / Early)\n",
        f"\nTotal patients: {len(y_true)} (TR-valid)",
        f"\n- pos: {int(y_true.sum())} ({int(y_true.sum())/len(y_true)*100:.1f}%)",
        f"\n- neg: {int((1-y_true).sum())} ({int((1-y_true).sum())/len(y_true)*100:.1f}%)",
        f"\n- Overall AUROC: {auroc:.4f}",
        f"\n- Optimal threshold (Youden's J): {optimal_threshold:.4f}",
        f"\n- Optimal Youden's J: {optimal_j:.4f}\n",
        f"\n## 예측 확률 분포",
        f"\n- min: {y_prob.min():.4f}, max: {y_prob.max():.4f}",
        f"\n- mean: {y_prob.mean():.4f} ± {y_prob.std():.4f}",
        f"\n- 5%: {np.percentile(y_prob, 5):.4f}, 50%: {np.percentile(y_prob, 50):.4f}, 95%: {np.percentile(y_prob, 95):.4f}\n",
        f"\n## 임계값별 메트릭 비교 (140 환자)\n",
        f"| Threshold | TP | TN | FP | FN | Recall_pos | Precision_pos | F1_pos | MCC | Bal Acc |",
        f"|---|---|---|---|---|---|---|---|---|---|",
    ]

    for label, m in [
        ('0.5 (baseline)', metrics_05),
        ('0.49', metrics_049),
        ('0.48', metrics_048),
        (f"{optimal_threshold:.4f} (Youden\\'s J)", metrics_opt),
        ('0.45', metrics_045),
    ]:
        md.append(f"| **{label}** | {m['TP']} | {m['TN']} | {m['FP']} | {m['FN']} | "
                  f"{m['recall_pos']:.3f} | {m['precision_pos']:.3f} | "
                  f"{m['f1_pos']:.3f} | {m['mcc']:.3f} | {m['balanced_acc']:.3f} |")

    md.append(f"\n\n## 진단군별 비교 (Youden's optimal vs baseline 0.5)\n")
    md.append(f"| 진단군 | n | n_pos | Baseline TP/FN | Baseline Recall_pos | Youden TP/FN | Youden Recall_pos |")
    md.append(f"|---|---|---|---|---|---|---|")
    for diag, d in per_diag.items():
        b = d['baseline_0.5']
        y = d['youden_optimal']
        md.append(f"| {diag} | {d['n']} | {d['n_pos']} | "
                  f"{b['TP']}/{b['FN']} | {b['recall_pos']:.3f} | "
                  f"{y['TP']}/{y['FN']} | {y['recall_pos']:.3f} |")

    md_str = '\n'.join(md)
    with open(out_dir / 'youden_analysis.md', 'w', encoding='utf-8') as f:
        f.write(md_str)
    print(f"  ✓ Saved: {out_dir}/youden_analysis.md")

    # ROC curve plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    # Youden's J optimal point
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
            label=f"Youden's J optimal\n(threshold={optimal_threshold:.3f})")
    # Baseline 0.5 point
    bl_idx = np.argmin(np.abs(thresholds - 0.5))
    ax.plot(fpr[bl_idx], tpr[bl_idx], 'g^', markersize=12,
            label=f'Baseline\n(threshold=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall_pos)', fontsize=11)
    ax.set_title(f'ROC Curve (B6 fold 0 / seed 42 / Early)\n140 TR-valid patients', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Prediction probability histogram by class
    ax = axes[1]
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ax.hist([neg_probs, pos_probs], bins=20, label=['Negative (n=105)', 'Positive (n=35)'],
            color=['#A23B72', '#2E86AB'], alpha=0.7, edgecolor='white')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, label='Baseline (0.5)')
    ax.axvline(x=optimal_threshold, color='red', linestyle='-', alpha=0.8,
               label=f"Youden's J ({optimal_threshold:.3f})")
    ax.set_xlabel('Predicted probability', fontsize=11)
    ax.set_ylabel('Patient count', fontsize=11)
    ax.set_title('Predicted probability distribution by true class', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve_youden.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_dir}/roc_curve_youden.png")

    print(f"\n{'='*70}")
    print(f"  ✓ Youden's J 분석 완료")
    print(f"{'='*70}\n")
    print(md_str[:2500])


if __name__ == '__main__':
    main()
