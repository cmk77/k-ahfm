import json

def L(p):
    return json.load(open(p))

cb1 = L('results/cblock1/cblock1_aggregated.json')   # B1 B2 B3
cb2 = L('results/cblock2/cblock2_aggregated.json')   # B4 B5 B6
DB  = {**cb1, **cb2}

def g(model, scen, group, metric, stat='mean'):
    try:
        return DB[f'{model}_{scen}'][group][metric][stat]
    except KeyError:
        return None

# ── 논문 PDF 보고값 ─────────────────────────────
# 표3a Early: AUROC, F1pos, MCC, BalAcc (mean±ci)
PAPER_3a = {
 'B1':(0.560,0.049, 0.352,0.049, 0.053,0.072, 0.527,0.034),
 'B2':(0.553,0.042, 0.318,0.055, 0.012,0.067, 0.504,0.035),
 'B3':(0.547,0.037, 0.354,0.048, 0.058,0.054, 0.524,0.026),
 'B4':(0.578,0.052, 0.337,0.055, 0.085,0.080, 0.536,0.039),
 'B5':(0.623,0.058, 0.326,0.076, 0.118,0.081, 0.555,0.042),
 'B6':(0.599,0.047, 0.319,0.062, 0.062,0.049, 0.532,0.024),
}
# 표3b Full
PAPER_3b = {
 'B1':(0.469,0.036, 0.281,0.061, -0.040,0.059, 0.485,0.028),
 'B2':(0.474,0.047, 0.249,0.066, -0.085,0.067, 0.459,0.036),
 'B3':(0.481,0.043, 0.318,0.057,  0.009,0.057, 0.506,0.027),
 'B4':(0.559,0.049, 0.293,0.054,  0.039,0.059, 0.511,0.024),
 'B5':(0.551,0.051, 0.254,0.075,  0.027,0.071, 0.510,0.034),
 'B6':(0.568,0.045, 0.308,0.061,  0.041,0.053, 0.523,0.028),
}
# 표4 진단: EarlyF1, EarlyAcc, FullF1, FullAcc (mean only, ci 별도)
PAPER_4 = {
 'B1':(0.515,0.554, 0.525,0.584),
 'B2':(0.537,0.570, 0.535,0.567),
 'B3':(0.502,0.536, 0.576,0.617),
 'B4':(0.720,0.771, 0.701,0.749),
 'B5':(0.771,0.800, 0.640,0.701),
 'B6':(0.660,0.710, 0.758,0.804),
}

def chk(name, paper, scen):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    print(f"{'M':3} {'metric':9} {'논문':>8} {'실측':>9} {'논문CI':>7} {'실측CI':>8}  판정")
    metrics = [('auroc','AUROC'),('f1_pos','F1pos'),('mcc','MCC'),('balanced_acc','BalAcc')]
    for m,(mp,cp,fp,cf,xp,cx,bp,cb) in paper.items():
        vals = [(mp,cp),(fp,cf),(xp,cx),(bp,cb)]
        for (key,disp),(pm,pc) in zip(metrics, vals):
            rm = g(m, scen, 'tr_metrics', key, 'mean')
            rc = g(m, scen, 'tr_metrics', key, 'ci95')
            if rm is None:
                print(f"{m:3} {disp:9} {pm:8.3f} {'없음':>9}"); continue
            ok = abs(rm-pm) < 0.0006
            okc = abs(rc-pc) < 0.0006
            flag = '✅' if ok else '❌'
            cflag = '' if okc else ' ⚠CI'
            print(f"{m:3} {disp:9} {pm:8.3f} {rm:9.4f} {pc:7.3f} {rc:8.4f}  {flag}{cflag}")

chk("표 3a — 주 과제 Early (TR)", PAPER_3a, 'early')
chk("표 3b — 주 과제 Full (TR)", PAPER_3b, 'full')

# 표4 진단
print(f"\n{'='*70}\n표 4 — 4-class 진단 (diag macro_f1 / accuracy)\n{'='*70}")
print(f"{'M':3} {'col':10} {'논문':>8} {'실측':>9}  판정")
for m,(ef,ea,ff,fa) in PAPER_4.items():
    pairs=[('EarlyF1','early','macro_f1',ef),('EarlyAcc','early','accuracy',ea),
           ('FullF1','full','macro_f1',ff),('FullAcc','full','accuracy',fa)]
    for disp,scen,key,pv in pairs:
        rv = g(m,scen,'diag_metrics',key,'mean')
        if rv is None: print(f"{m:3} {disp:10} {pv:8.3f} {'없음':>9}"); continue
        ok=abs(rv-pv)<0.0006
        print(f"{m:3} {disp:10} {pv:8.3f} {rv:9.4f}  {'✅' if ok else '❌'}")
