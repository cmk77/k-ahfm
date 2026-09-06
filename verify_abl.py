import json, numpy as np

abl = json.load(open('results/hyperedge_ablation_v2/ablation_v2_aggregated.json'))

def a(cond, scen, metric, stat='mean'):
    return abl[f'{cond}_{scen}'][metric][stat]

# ── 표 6a/6b 검증 ──────────────────────────────
# 논문 PDF: (TR_AUROC, F1pos, MCC, DiagF1, ΔAUROC)
PAPER_6a = {  # Early
 'full_B6':        (0.557,0.261, 0.001,0.656, None),
 'minus_E_trajectory':       (0.555,0.256,-0.000,0.671,-0.002),
 'minus_E_co-trajectory':    (0.543,0.225,-0.026,0.627,-0.015),
 'minus_E_intervention-response':(0.558,0.252,-0.027,0.644,+0.001),
 'minus_E_clinical_prior':   (0.541,0.292, 0.025,0.632,-0.016),
}
PAPER_6b = {  # Full
 'full_B6':        (0.528,0.279, 0.013,0.663, None),
 'minus_E_trajectory':       (0.555,0.295, 0.044,0.704,+0.026),
 'minus_E_co-trajectory':    (0.518,0.262, 0.000,0.698,-0.011),
 'minus_E_intervention-response':(0.536,0.262, 0.007,0.635,+0.008),
 'minus_E_clinical_prior':   (0.550,0.268, 0.046,0.675,+0.021),
}

def chk6(name, paper, scen):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print(f"{'condition':26} {'metric':8} {'논문':>7} {'실측':>9}  판정")
    base = a('full_B6', scen, 'tr_auroc', 'mean')
    keymap = [('tr_auroc','AUROC'),('tr_f1_pos','F1pos'),('tr_mcc','MCC'),('diag_macro_f1','DiagF1')]
    for cond,(au,f1,mcc,df,dlt) in paper.items():
        for (k,disp),pv in zip(keymap,[au,f1,mcc,df]):
            try: rv=a(cond,scen,k,'mean')
            except KeyError: print(f"{cond:26} {disp:8} {pv:7.3f} {'키없음':>9}"); continue
            print(f"{cond:26} {disp:8} {pv:7.3f} {rv:9.4f}  {'✅' if abs(rv-pv)<0.0006 else '❌'}")
        # ΔAUROC 검증
        if dlt is not None:
            rdlt = a(cond,scen,'tr_auroc','mean') - base
            print(f"{cond:26} {'ΔAUROC':8} {dlt:+7.3f} {rdlt:+9.4f}  {'✅' if abs(rdlt-dlt)<0.0011 else '❌'}")

chk6("표 6a — Ablation Early", PAPER_6a, 'early')
chk6("표 6b — Ablation Full",  PAPER_6b, 'full')

# ── §4.5 α_e 가중치 검증 ───────────────────────
print(f"\n{'='*72}\n§4.5 — α_e hyperedge 가중치 (case_summary_table.json)\n{'='*72}")
cases = json.load(open('results/b6_attention_fold0_seed42_early/case_summary_table.json'))
print(f"  case 수: {len(cases)} (논문 '9 cases')")
print(f"  첫 case keys: {list(cases[0].keys())}")
# alpha 관련 키 자동 탐색
ak = [k for k in cases[0] if 'alpha' in k.lower() or any(e in k.lower() for e in ['traj','interv','clinic','co_'])]
print(f"  α 관련 키: {ak}")
# 각 edge type 평균 계산
import collections
sums = collections.defaultdict(list)
for c in cases:
    for k in ak:
        if isinstance(c[k],(int,float)): sums[k].append(c[k])
print("\n  edge type별 평균 (논문: co_traj 2.128 > traj 1.381 > clinical 1.301 > interv 0.708):")
for k,v in sums.items():
    print(f"    {k:30} mean={np.mean(v):.4f}  (n={len(v)})")

# ── §4.5 run 성능 검증 (test AUROC 0.626, diag F1 0.855) ──
print(f"\n{'='*72}\n§4.5 run 성능 (train_summary.json)\n{'='*72}")
ts = json.load(open('results/b6_attention_fold0_seed42_early/train_summary.json'))
print(f"  test_tr_metrics: {ts.get('test_tr_metrics')}")
print(f"  test_diag_metrics: {ts.get('test_diag_metrics')}")
print(f"  논문 §4.5: test AUROC 0.626, 진단 macro F1 0.855")
