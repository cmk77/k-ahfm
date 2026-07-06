import json, numpy as np, collections

abl = json.load(open('results/hyperedge_ablation_v2/ablation_v2_aggregated.json'))
KEYS = list(abl.keys())

def find(cond_hint, scen):
    # cond_hint의 핵심 토큰으로 실제 키 매칭
    toks = cond_hint.replace('minus_E_','').replace('_','').replace('-','').lower()
    for k in KEYS:
        if not k.endswith('_'+scen): continue
        norm = k.replace('_'+scen,'').replace('minus_E_','').replace('_','').replace('-','').lower()
        if norm == toks: return k
    return None

def a(realkey, metric, stat='mean'):
    return abl[realkey][metric][stat]

PAPER_6a = {
 'full_B6':(0.557,0.261,0.001,0.656,None),
 'trajectory':(0.555,0.256,-0.000,0.671,-0.002),
 'cotrajectory':(0.543,0.225,-0.026,0.627,-0.015),
 'interventionresponse':(0.558,0.252,-0.027,0.644,+0.001),
 'clinicalprior':(0.541,0.292,0.025,0.632,-0.016)}
PAPER_6b = {
 'full_B6':(0.528,0.279,0.013,0.663,None),
 'trajectory':(0.555,0.295,0.044,0.704,+0.026),
 'cotrajectory':(0.518,0.262,0.000,0.698,-0.011),
 'interventionresponse':(0.536,0.262,0.007,0.635,+0.008),
 'clinicalprior':(0.550,0.268,0.046,0.675,+0.021)}

def chk6(name, paper, scen):
    print("\n"+"="*72+"\n"+name+"\n"+"="*72)
    print(f"{'condition':24} {'metric':8} {'논문':>7} {'실측':>9}  판정")
    base_key = 'full_B6_'+scen
    base = abl[base_key]['tr_auroc']['mean']
    keymap=[('tr_auroc','AUROC'),('tr_f1_pos','F1pos'),('tr_mcc','MCC'),('diag_macro_f1','DiagF1')]
    for cond,(au,f1,mcc,df,dlt) in paper.items():
        rk = (cond+'_'+scen) if cond=='full_B6' else find(cond,scen)
        if rk is None: print(f"{cond:24} (키 못찾음)"); continue
        for (k,disp),pv in zip(keymap,[au,f1,mcc,df]):
            rv=abl[rk][k]['mean']
            print(f"{cond:24} {disp:8} {pv:7.3f} {rv:9.4f}  {'✅' if abs(rv-pv)<0.0006 else '❌'}")
        if dlt is not None:
            rd=abl[rk]['tr_auroc']['mean']-base
            print(f"{cond:24} {'ΔAUROC':8} {dlt:+7.3f} {rd:+9.4f}  {'✅' if abs(rd-dlt)<0.0011 else '❌'}")

chk6("표 6a — Ablation Early", PAPER_6a, 'early')
chk6("표 6b — Ablation Full",  PAPER_6b, 'full')

# §4.5 α_e
print("\n"+"="*72+"\n§4.5 — α_e 가중치 + run 성능\n"+"="*72)
cases=json.load(open('results/b6_attention_fold0_seed42_early/case_summary_table.json'))
print(f"  case 수={len(cases)}, keys={list(cases[0].keys())}")
ak=[k for k in cases[0] if any(e in k.lower() for e in ['alpha','traj','interv','clinic','co_','edge'])]
print(f"  α 키: {ak}")
agg=collections.defaultdict(list)
for c in cases:
    for k in ak:
        if isinstance(c[k],(int,float)): agg[k].append(c[k])
print("  논문: co_traj 2.128 > traj 1.381 > clinical 1.301 > interv 0.708")
for k in ak:
    if agg[k]: print(f"    {k:28} mean={np.mean(agg[k]):.4f} (n={len(agg[k])})")

ts=json.load(open('results/b6_attention_fold0_seed42_early/train_summary.json'))
print(f"\n  run 성능 (논문 test AUROC 0.626 / diag F1 0.855):")
print(f"    test_tr_metrics  = {ts.get('test_tr_metrics')}")
print(f"    test_diag_metrics= {ts.get('test_diag_metrics')}")
