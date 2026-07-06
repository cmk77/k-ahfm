#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13c_youden_oof_multiseed.py — 5-seed × 5-fold OOF Youden 집계

각 seed s에 대해: fold0~4의 test 예측만 모아 140명 OOF 구성
→ AUROC, Youden threshold, threshold별 metric, 진단군별 recall 산출.
그 뒤 5 seed의 평균 ± 95% CI 보고. (leakage 없음 = 진짜 out-of-fold)

사용법:
    python 13c_youden_oof_multiseed.py \
        --root <repo>/results \
        --pattern b6_oof_fold{F}_seed{S}_early \
        --fold0-seed42 <repo>/results/b6_attention_fold0_seed42_early \
        --seeds 0 42 2026 7 1024 \
        --output <repo>/results/youden_oof_multiseed
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef

DIAGS = ['depression', 'anxiety', 'addiction']


def load_test(pred_dir):
    p = Path(pred_dir) / 'patient_predictions_test.json'
    if not p.exists():
        raise FileNotFoundError(f"없음: {p}")
    with open(p) as f:
        preds = json.load(f)
    return [x for x in preds if x.get('tr_valid')]


def metrics_at(y_true, y_prob, thr):
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0
    return dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                recall_pos=rec, precision_pos=prec, f1_pos=f1,
                mcc=mcc, bal_acc=(rec + rec_neg) / 2)


def diag_recall(preds, thr):
    out = {}
    for dn in DIAGS:
        sub = [p for p in preds if p['diagnosis'] == dn]
        yt = np.array([p['true_tr_label'] for p in sub])
        yp = np.array([p['pred_tr_prob'] for p in sub])
        npos = int(yt.sum())
        tp = int(((yt == 1) & (yp >= thr)).sum())
        out[dn] = tp / npos if npos else np.nan
    return out


def mean_ci(vals):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return {'mean': None, 'ci95': None, 'n': 0}
    m = a.mean()
    ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {'mean': round(float(m), 4), 'ci95': round(float(ci), 4), 'n': len(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--pattern', default='b6_oof_fold{F}_seed{S}_early')
    ap.add_argument('--fold0-seed42', default=None,
                    help='fold0/seed42만 기존 디렉토리명이 다를 때 지정')
    ap.add_argument('--seeds', nargs='+', default=['0', '42', '2026', '7', '1024'])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.output).expanduser(); out.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    aurocs, yj_thrs = [], []
    base_metrics = {k: [] for k in ['recall_pos', 'precision_pos', 'f1_pos', 'mcc', 'bal_acc']}
    yj_metrics = {k: [] for k in base_metrics}
    diag_base = {d: [] for d in DIAGS}
    diag_yj = {d: [] for d in DIAGS}

    for S in args.seeds:
        preds = []
        for F in range(5):
            if F == 0 and S == '42' and args.fold0_seed42:
                d = Path(args.fold0_seed42).expanduser()
            else:
                d = root / args.pattern.format(F=F, S=S)
            preds += load_test(d)
        ids = [p['patient_id'] for p in preds]
        assert len(ids) == len(set(ids)), f"seed {S}: 환자 중복"
        y_true = np.array([p['true_tr_label'] for p in preds])
        y_prob = np.array([p['pred_tr_prob'] for p in preds])

        auroc = roc_auc_score(y_true, y_prob)
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        yj = float(thr[np.argmax(tpr - fpr)])
        aurocs.append(auroc); yj_thrs.append(yj)

        mb = metrics_at(y_true, y_prob, 0.5)
        my = metrics_at(y_true, y_prob, yj)
        for k in base_metrics:
            base_metrics[k].append(mb[k]); yj_metrics[k].append(my[k])
        db, dy = diag_recall(preds, 0.5), diag_recall(preds, yj)
        for dn in DIAGS:
            diag_base[dn].append(db[dn]); diag_yj[dn].append(dy[dn])

        per_seed[S] = dict(n=len(preds), n_pos=int(y_true.sum()),
                           auroc=round(auroc, 4), youden=round(yj, 4),
                           prob_mean=round(float(y_prob.mean()), 4),
                           prob_std=round(float(y_prob.std()), 4),
                           baseline=mb, youden_metrics=my,
                           diag_base=db, diag_youden=dy)
        print(f"seed {S:>5}: AUROC {auroc:.4f}  YJ_thr {yj:.4f}  "
              f"base_recall {mb['recall_pos']:.3f}  YJ_recall {my['recall_pos']:.3f}  "
              f"YJ_MCC {my['mcc']:.3f}")

    print("\n===== 5-seed 집계 (mean ± 95% CI) =====")
    print(f"AUROC (OOF):        {mean_ci(aurocs)}")
    print(f"Youden threshold:   {mean_ci(yj_thrs)}")
    print("\n[baseline 0.5]")
    for k in base_metrics:
        print(f"  {k:>14}: {mean_ci(base_metrics[k])}")
    print("\n[Youden optimal]")
    for k in yj_metrics:
        print(f"  {k:>14}: {mean_ci(yj_metrics[k])}")
    print("\n[진단군별 recall: baseline → Youden]")
    for dn in DIAGS:
        print(f"  {dn:>11}: base {mean_ci(diag_base[dn])}  →  YJ {mean_ci(diag_yj[dn])}")

    summary = dict(
        seeds=args.seeds, per_seed=per_seed,
        agg=dict(auroc=mean_ci(aurocs), youden_threshold=mean_ci(yj_thrs),
                 baseline={k: mean_ci(v) for k, v in base_metrics.items()},
                 youden={k: mean_ci(v) for k, v in yj_metrics.items()},
                 diag_baseline={d: mean_ci(diag_base[d]) for d in DIAGS},
                 diag_youden={d: mean_ci(diag_yj[d]) for d in DIAGS}))
    with open(out / 'youden_oof_multiseed.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out / 'youden_oof_multiseed.json'}")


if __name__ == '__main__':
    main()
