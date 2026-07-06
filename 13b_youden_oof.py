#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13b_youden_oof.py — Out-of-fold Youden 재분석 (leakage 제거판)

기존 13_youden_threshold.py는 단일 fold0 run의 train+val+test 예측을
합쳐 140명을 구성했음 → train 101명이 모델 학습 데이터이므로 leakage.

본 스크립트는 5개 fold 각각의 *test 예측만* 모아 140명을 구성한다.
환자 단위 5-fold이므로 5×28=140, 중복 없음 = 진짜 out-of-fold.

사용법:
    python 13b_youden_oof.py \
        --fold0 <repo>/results/b6_attention_fold0_seed42_early \
        --oof-root <repo>/results \
        --oof-pattern b6_oof_fold{F}_seed42_early \
        --output <repo>/results/youden_oof_seed42
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef


def load_test_only(pred_dir):
    """한 fold 디렉토리에서 test split 예측만 로드 (TR-valid만)."""
    p = Path(pred_dir) / 'patient_predictions_test.json'
    if not p.exists():
        raise FileNotFoundError(f"  ✗ {p} 없음")
    with open(p) as f:
        preds = json.load(f)
    valid = [x for x in preds if x.get('tr_valid')]
    print(f"  + {pred_dir.name if hasattr(pred_dir,'name') else pred_dir}: "
          f"test {len(preds)}명 중 TR-valid {len(valid)}명")
    return valid


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    f1_pos = (2 * prec_pos * recall_pos / (prec_pos + recall_pos)
              if (prec_pos + recall_pos) else 0.0)
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (recall_pos + recall_neg) / 2
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0
    return {'threshold': round(float(threshold), 4),
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'recall_pos': round(recall_pos, 3), 'precision_pos': round(prec_pos, 3),
            'f1_pos': round(f1_pos, 3), 'mcc': round(mcc, 3), 'bal_acc': round(bal_acc, 3)}


def per_diag(preds, threshold):
    out = {}
    for dn in ['depression', 'anxiety', 'addiction']:
        sub = [p for p in preds if p['diagnosis'] == dn]
        if not sub:
            continue
        yt = [p['true_tr_label'] for p in sub]
        yp = [p['pred_tr_prob'] for p in sub]
        ypred = (np.asarray(yp) >= threshold).astype(int)
        n_pos = int(sum(yt))
        tp = int(((np.asarray(yt) == 1) & (ypred == 1)).sum())
        out[dn] = {'n': len(sub), 'n_pos': n_pos, 'TP': tp,
                   'FN': n_pos - tp,
                   'recall': round(tp / n_pos, 3) if n_pos else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold0', required=True, help='기존 fold0 결과 디렉토리')
    ap.add_argument('--oof-root', required=True, help='fold1~4 상위 디렉토리')
    ap.add_argument('--oof-pattern', default='b6_oof_fold{F}_seed42_early')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    root = Path(args.oof_root).expanduser()
    out = Path(args.output).expanduser(); out.mkdir(parents=True, exist_ok=True)

    print("[1/4] OOF test 예측 로드 (fold별 test만)")
    preds = load_test_only(Path(args.fold0).expanduser())
    for F in [1, 2, 3, 4]:
        preds += load_test_only(root / args.oof_pattern.format(F=F))

    # 환자 중복 검사
    ids = [p['patient_id'] for p in preds]
    dup = len(ids) - len(set(ids))
    print(f"  → OOF 합계 {len(preds)}명 (중복 {dup}명)")
    assert dup == 0, "환자 중복 발생 — fold 분할 확인 필요"

    y_true = np.array([p['true_tr_label'] for p in preds])
    y_prob = np.array([p['pred_tr_prob'] for p in preds])
    print(f"  → 양성(호전) {int(y_true.sum())}명 / 전체 {len(y_true)}명")
    print(f"  → pred_prob: min {y_prob.min():.4f}  max {y_prob.max():.4f}  "
          f"mean {y_prob.mean():.4f} ± {y_prob.std():.4f}")

    print("\n[2/4] ROC + Youden's J")
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    j = tpr - fpr
    opt_thr = float(thr[np.argmax(j)])
    print(f"  Overall AUROC (OOF, 140): {auroc:.4f}")
    print(f"  Youden optimal threshold: {opt_thr:.4f}  (J={j.max():.4f})")

    print("\n[3/4] threshold별 metric")
    rows = {'0.5': compute_metrics(y_true, y_prob, 0.5),
            f'youden_{opt_thr:.4f}': compute_metrics(y_true, y_prob, opt_thr)}
    for t in (0.49, 0.48, 0.45):
        rows[str(t)] = compute_metrics(y_true, y_prob, t)
    for k, v in rows.items():
        print(f"  {k:>16}: " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()
                                          if kk not in ('threshold',)))

    print("\n[4/4] 진단군별 (baseline 0.5 vs Youden)")
    pd_base = per_diag(preds, 0.5)
    pd_yj = per_diag(preds, opt_thr)
    for dn in ['depression', 'anxiety', 'addiction']:
        b, y = pd_base.get(dn, {}), pd_yj.get(dn, {})
        print(f"  {dn:>11}: n={b.get('n')} n_pos={b.get('n_pos')}  "
              f"base recall {b.get('recall')} ({b.get('TP')}/{b.get('n_pos')})  →  "
              f"YJ recall {y.get('recall')} ({y.get('TP')}/{y.get('n_pos')})")

    result = {'auroc_oof': round(auroc, 4), 'youden_threshold': round(opt_thr, 4),
              'n_total': len(y_true), 'n_pos': int(y_true.sum()),
              'prob_min': round(float(y_prob.min()), 4),
              'prob_max': round(float(y_prob.max()), 4),
              'prob_mean': round(float(y_prob.mean()), 4),
              'prob_std': round(float(y_prob.std()), 4),
              'thresholds': rows,
              'per_diag_baseline': pd_base, 'per_diag_youden': pd_yj}
    with open(out / 'youden_oof_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out / 'youden_oof_analysis.json'}")


if __name__ == '__main__':
    main()
