import numpy as np, collections

TYPE = {0:'E_trajectory', 1:'E_co-trajectory', 2:'E_intervention', 3:'E_clinical_prior'}
PAPER = {'E_co-trajectory':2.128, 'E_trajectory':1.381, 'E_clinical_prior':1.301, 'E_intervention':0.708}

d = np.load('results/b6_attention_fold0_seed42_early/patient_attention_test.npz', allow_pickle=True)
pids = sorted(set(k.split('__')[0] for k in d.files if k != '__patient_ids'))
print(f"test 환자 수: {len(pids)}")

def type_means(arr_key):
    agg = collections.defaultdict(list)
    for pid in pids:
        a = d[f'{pid}__{arr_key}']
        et = d[f'{pid}__edge_types']
        em = d[f'{pid}__edge_mask']
        a, et = a[em], et[em]
        for t in set(et.tolist()):
            agg[TYPE[t]].append(a[et==t].mean())
    return agg

print("\n=== alpha_e1 (raw) edge type별 — 환자평균의 평균 ===")
print(f"{'type':20} {'논문':>7} {'실측':>9}  판정")
agg = type_means('alpha_e1')
order = []
for t in ['E_co-trajectory','E_trajectory','E_clinical_prior','E_intervention']:
    m = float(np.mean(agg[t]))
    order.append((t, m))
    print(f"{t:20} {PAPER[t]:7.3f} {m:9.4f}  {'OK' if abs(m-PAPER[t])<0.05 else 'CHK'}")

ranked = sorted(order, key=lambda x: -x[1])
rank_str = ' > '.join(t.split('_')[-1] + '(' + format(m, '.2f') + ')' for t, m in ranked)
print("\n실측 순위:", rank_str)
print("논문 순위: co-trajectory(2.13) > trajectory(1.38) > clinical_prior(1.30) > intervention(0.71)")

print("\n=== alpha_e2 (renormalized) 동일 계산 ===")
agg2 = type_means('alpha_e2')
for t in ['E_co-trajectory','E_trajectory','E_clinical_prior','E_intervention']:
    print(f"  {t:20} {float(np.mean(agg2[t])):.4f}")

print("\n=== beta_v 범위 (논문 0.330-0.337) ===")
allb = []
for pid in pids:
    allb.extend(d[f'{pid}__beta_v'].tolist())
allb = np.array(allb)
print(f"  beta_v: min={allb.min():.4f}, max={allb.max():.4f}, mean={allb.mean():.4f}")
