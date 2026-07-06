#!/usr/bin/env python
"""
11_select_and_visualize.py 패치 — 시각화 버그 수정 + 안전 저장.

문제: case마다 edge type 개수가 다를 수 있어 hyperedge_summary 차트에서
      길이 불일치 에러 (ValueError: shape mismatch (9,) vs (7,))

해결: 모든 edge type을 수집 후 missing은 0 또는 nan으로 채움.

추가: selected_cases.json + case_summary_table.json이 에러 전에 안 저장된
      경우 — 다시 실행 시 정상 저장.

사용법: 기존 11_select_and_visualize.py를 이 파일로 교체 후 재실행:
    cp 11_select_and_visualize_v2.py 11_select_and_visualize.py
    python 11_select_and_visualize.py --output <repo>/results/b6_attention_fold0_seed42_early
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False
try:
    for font in ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR']:
        from matplotlib import font_manager
        if any(font in f.name for f in font_manager.fontManager.ttflist):
            mpl.rcParams['font.family'] = font
            break
except Exception:
    pass

DIAGNOSIS_NAMES = {0: 'depression', 1: 'anxiety', 2: 'addiction', 3: 'other'}
DIAGNOSIS_DISPLAY = {'depression': 'Depression', 'anxiety': 'Anxiety', 'addiction': 'Addiction'}
OUTCOME_LABELS = ['TP', 'TN', 'FP', 'FN']
OUTCOME_DISPLAY = {
    'TP': 'True Positive (정답 호전)',
    'TN': 'True Negative (정답 비호전)',
    'FP': 'False Positive (오답 호전)',
    'FN': 'False Negative (오답 비호전)',
}
EDGE_TYPE_NAMES = {
    0: 'E_trajectory',
    1: 'E_co-trajectory',
    2: 'E_intervention',
    3: 'E_clinical_prior',
    4: 'E_affective_pattern',
}


def classify_outcome(true_tr, pred_tr_prob, threshold=0.5):
    if true_tr is None:
        return None
    pred = 1 if pred_tr_prob > threshold else 0
    if true_tr == 1 and pred == 1: return 'TP'
    if true_tr == 0 and pred == 0: return 'TN'
    if true_tr == 0 and pred == 1: return 'FP'
    if true_tr == 1 and pred == 0: return 'FN'
    return None


def select_cases_from_predictions(predictions):
    by_group = defaultdict(list)
    for p in predictions:
        if not p.get('tr_valid'):
            continue
        diag = p['diagnosis']
        if diag not in ('depression', 'anxiety', 'addiction'):
            continue
        outcome = classify_outcome(p['true_tr_label'], p['pred_tr_prob'])
        if outcome is None:
            continue
        by_group[(diag, outcome)].append(p)

    selected = []
    for diag in ['depression', 'anxiety', 'addiction']:
        for outcome in OUTCOME_LABELS:
            cands = by_group.get((diag, outcome), [])
            if not cands:
                selected.append({
                    'diagnosis': diag,
                    'outcome': outcome,
                    'patient_id': None,
                    'true_tr_label': None,
                    'pred_tr_prob': None,
                    'note': f'No case found for {diag}/{outcome}',
                })
                continue
            if outcome == 'TP':
                cands.sort(key=lambda x: -x['pred_tr_prob'])
            elif outcome == 'TN':
                cands.sort(key=lambda x: x['pred_tr_prob'])
            elif outcome == 'FP':
                cands.sort(key=lambda x: -x['pred_tr_prob'])
            elif outcome == 'FN':
                cands.sort(key=lambda x: x['pred_tr_prob'])
            best = cands[0]
            selected.append({
                'diagnosis': diag,
                'outcome': outcome,
                'patient_id': best['patient_id'],
                'true_tr_label': best['true_tr_label'],
                'pred_tr_prob': round(best['pred_tr_prob'], 4),
                'pred_tr_label': best['pred_tr_label'],
                'num_nodes': best['num_nodes'],
                'true_diag_label': best['true_diag_label'],
                'pred_diag_label': best['pred_diag_label'],
                'n_candidates': len(cands),
            })
    return selected


def visualize_case(case, attention_data, cases_dir):
    pid = case['patient_id']
    if pid is None:
        return None

    if f'{pid}__alpha_e1' not in attention_data:
        print(f"  ⚠ No attention data for {pid}")
        return None

    alpha_e1 = attention_data[f'{pid}__alpha_e1']
    alpha_e2 = attention_data[f'{pid}__alpha_e2']
    beta_v = attention_data[f'{pid}__beta_v']
    edge_types = attention_data[f'{pid}__edge_types']
    node_mask = attention_data[f'{pid}__node_mask'].astype(bool)
    edge_mask = attention_data[f'{pid}__edge_mask'].astype(bool)

    valid_beta = beta_v[node_mask]
    valid_alpha_e1 = alpha_e1[edge_mask]
    valid_alpha_e2 = alpha_e2[edge_mask]
    valid_edge_types = edge_types[edge_mask]

    edge_type_avg_l1 = {}
    edge_type_avg_l2 = {}
    edge_type_count = {}
    for et, w1, w2 in zip(valid_edge_types, valid_alpha_e1, valid_alpha_e2):
        et = int(et)
        edge_type_avg_l1.setdefault(et, []).append(float(w1))
        edge_type_avg_l2.setdefault(et, []).append(float(w2))
        edge_type_count[et] = edge_type_count.get(et, 0) + 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = (f"{case['diagnosis'].upper()} / {case['outcome']}  "
             f"(Patient {pid})\n"
             f"True TR={case['true_tr_label']}, "
             f"Pred prob={case['pred_tr_prob']:.3f} → {case['pred_tr_label']}, "
             f"Nodes={case['num_nodes']}")
    fig.suptitle(title, fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.bar(range(len(valid_beta)), valid_beta, color='steelblue', alpha=0.7)
    ax.set_xlabel('Node index')
    ax.set_ylabel('beta_v (attention pooling weight)')
    ax.set_title(f'(a) Node attention beta_v ({len(valid_beta)} valid nodes)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0/len(valid_beta), color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    types_sorted = sorted(edge_type_avg_l1.keys())
    means_l1 = [np.mean(edge_type_avg_l1[t]) for t in types_sorted]
    stds_l1 = [np.std(edge_type_avg_l1[t]) for t in types_sorted]
    counts = [edge_type_count[t] for t in types_sorted]
    type_labels = [f'{EDGE_TYPE_NAMES.get(t, f"Type{t}")}\n(n={c})' for t, c in zip(types_sorted, counts)]
    bars = ax.bar(range(len(types_sorted)), means_l1, yerr=stds_l1, capsize=4,
                  color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95'][:len(types_sorted)],
                  alpha=0.75)
    ax.set_xticks(range(len(types_sorted)))
    ax.set_xticklabels(type_labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Mean alpha_e (Layer 1)')
    ax.set_title('(b) Hyperedge weights alpha_e (Layer 1)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means_l1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 0]
    means_l2 = [np.mean(edge_type_avg_l2[t]) for t in types_sorted]
    stds_l2 = [np.std(edge_type_avg_l2[t]) for t in types_sorted]
    bars = ax.bar(range(len(types_sorted)), means_l2, yerr=stds_l2, capsize=4,
                  color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95'][:len(types_sorted)],
                  alpha=0.75)
    ax.set_xticks(range(len(types_sorted)))
    ax.set_xticklabels(type_labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Mean alpha_e (Layer 2)')
    ax.set_title('(c) Hyperedge weights alpha_e (Layer 2)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means_l2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 1]
    colors_by_type = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
    point_colors = [colors_by_type[int(t) % 5] for t in valid_edge_types]
    ax.scatter(range(len(valid_alpha_e2)), valid_alpha_e2, c=point_colors, alpha=0.7, s=40)
    ax.set_xlabel('Hyperedge index')
    ax.set_ylabel('alpha_e (Layer 2)')
    ax.set_title(f'(d) Individual hyperedge weights (Layer 2, total {len(valid_alpha_e2)} edges)')
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_by_type[t], label=EDGE_TYPE_NAMES.get(t, f'Type{t}'))
                       for t in types_sorted]
    ax.legend(handles=legend_elements, fontsize=8, loc='best')

    plt.tight_layout()
    case_label = f"{case['diagnosis']}_{case['outcome']}_{pid}"
    out_path = cases_dir / f"{case_label}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

    return {
        'case_label': case_label,
        'png_path': str(out_path.name),
        'edge_type_means_l2': {EDGE_TYPE_NAMES.get(t, f'Type{t}'): float(np.mean(edge_type_avg_l2[t]))
                               for t in types_sorted},
        'edge_type_means_l1': {EDGE_TYPE_NAMES.get(t, f'Type{t}'): float(np.mean(edge_type_avg_l1[t]))
                               for t in types_sorted},
        'edge_type_counts': {EDGE_TYPE_NAMES.get(t, f'Type{t}'): int(edge_type_count[t])
                             for t in types_sorted},
        'beta_v_top3': [(int(i), float(v)) for i, v in
                        sorted(enumerate(valid_beta), key=lambda x: -x[1])[:3]],
        'beta_v_all': [float(v) for v in valid_beta],
        'n_valid_nodes': int(node_mask.sum()),
        'n_valid_edges': int(edge_mask.sum()),
    }


def make_hyperedge_summary(selected_with_stats, output_dir):
    """[FIXED] 모든 edge type 통일적으로 채우기 (missing → 0)."""
    # Phase 1: collect ALL edge type names across all cases
    all_edge_names = set()
    valid_cases = []
    for s in selected_with_stats:
        if s.get('vis_stats') is None:
            continue
        valid_cases.append(s)
        all_edge_names.update(s['vis_stats']['edge_type_means_l2'].keys())

    if not valid_cases:
        print(f"  ⚠ No valid cases for summary plot")
        return

    edge_names_sorted = sorted(all_edge_names)
    print(f"  All edge types found across cases: {edge_names_sorted}")

    # Phase 2: build case x edge_type matrix (missing → 0)
    case_labels = []
    edge_means_matrix = {name: [] for name in edge_names_sorted}
    for s in valid_cases:
        case_labels.append(f"{s['diagnosis'][:3]}_{s['outcome']}")
        for et_name in edge_names_sorted:
            val = s['vis_stats']['edge_type_means_l2'].get(et_name, 0.0)
            edge_means_matrix[et_name].append(val)

    # All lists must now have same length
    n_cases = len(case_labels)
    for name, vals in edge_means_matrix.items():
        assert len(vals) == n_cases, f"Length mismatch: {name} has {len(vals)} != {n_cases}"

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    n_edges = len(edge_names_sorted)
    x = np.arange(n_cases)
    width = 0.18 if n_edges <= 4 else 0.15

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
    for i, et_name in enumerate(edge_names_sorted):
        offset = (i - (n_edges - 1) / 2) * width
        ax.bar(x + offset, edge_means_matrix[et_name], width,
               label=et_name, color=colors[i % 5], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Mean alpha_e (Layer 2)')
    ax.set_title('12 cases — Hyperedge weights by type (Layer 2)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperedge_summary.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ hyperedge_summary.png saved")


def make_outcome_distribution_plot(predictions, output_dir):
    """[NEW] outcome 분포 시각화 (12 cases 매트릭스)."""
    matrix = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        if not p.get('tr_valid'):
            continue
        diag = p['diagnosis']
        if diag not in ('depression', 'anxiety', 'addiction'):
            continue
        outcome = classify_outcome(p['true_tr_label'], p['pred_tr_prob'])
        if outcome:
            matrix[diag][outcome] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    diagnoses = ['depression', 'anxiety', 'addiction']
    outcomes = ['TP', 'TN', 'FP', 'FN']
    data = np.array([[matrix[d][o] for o in outcomes] for d in diagnoses])

    im = ax.imshow(data, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcomes)
    ax.set_yticks(range(len(diagnoses)))
    ax.set_yticklabels([d.title() for d in diagnoses])
    for i in range(len(diagnoses)):
        for j in range(len(outcomes)):
            ax.text(j, i, str(data[i, j]), ha='center', va='center',
                    color='white' if data[i, j] > data.max()/2 else 'black',
                    fontsize=13, fontweight='bold')
    ax.set_title('Outcome distribution by diagnosis (140 TR-valid patients)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'outcome_distribution.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ outcome_distribution.png saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--use-splits', nargs='+', default=['test', 'val', 'train'])
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    cases_dir = output_dir / 'cases'
    cases_dir.mkdir(exist_ok=True)

    print(f"[1/5] Loading predictions...")
    all_predictions = []
    all_attention = {}
    for split in args.use_splits:
        pred_path = output_dir / f'patient_predictions_{split}.json'
        attn_path = output_dir / f'patient_attention_{split}.npz'
        if not pred_path.exists() or not attn_path.exists():
            print(f"  ⚠ {split} files not found, skipping")
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        attn_npz = np.load(attn_path, allow_pickle=True)
        attn_dict = {k: attn_npz[k] for k in attn_npz.files}
        all_predictions.extend(preds)
        all_attention.update(attn_dict)
        print(f"  + {split}: {len(preds)} patients")
    print(f"  Total: {len(all_predictions)} patients")

    print(f"\n[2/5] Outcome distribution:")
    outcome_counts = defaultdict(int)
    for p in all_predictions:
        if not p.get('tr_valid'):
            continue
        diag = p['diagnosis']
        outcome = classify_outcome(p['true_tr_label'], p['pred_tr_prob'])
        if outcome and diag in ('depression', 'anxiety', 'addiction'):
            outcome_counts[(diag, outcome)] += 1
    for diag in ['depression', 'anxiety', 'addiction']:
        for o in OUTCOME_LABELS:
            print(f"  {diag:>10s} / {o}: {outcome_counts.get((diag, o), 0)}")

    print(f"\n[3/5] Selecting 12 cases...")
    selected = select_cases_from_predictions(all_predictions)
    for s in selected:
        if s['patient_id']:
            print(f"  {s['diagnosis']:>10s} / {s['outcome']}: {s['patient_id']} "
                  f"(true_tr={s['true_tr_label']}, pred_prob={s['pred_tr_prob']:.3f}, "
                  f"nodes={s['num_nodes']}, n_cands={s['n_candidates']})")
        else:
            print(f"  {s['diagnosis']:>10s} / {s['outcome']}: NO CASE")

    print(f"\n[4/5] Visualizing cases...")
    selected_with_stats = []
    for case in selected:
        if case['patient_id'] is None:
            case['vis_stats'] = None
            selected_with_stats.append(case)
            continue
        stats = visualize_case(case, all_attention, cases_dir)
        case['vis_stats'] = stats
        selected_with_stats.append(case)
        if stats:
            print(f"  ✓ {stats['case_label']}.png")

    # FIXED: hyperedge_summary + 추가 outcome plot
    print(f"\n[5/5] Summary plots + save metadata...")
    try:
        make_hyperedge_summary(selected_with_stats, output_dir)
    except Exception as e:
        print(f"  ⚠ hyperedge_summary failed: {e}")
    try:
        make_outcome_distribution_plot(all_predictions, output_dir)
    except Exception as e:
        print(f"  ⚠ outcome_distribution failed: {e}")

    # ALWAYS save metadata (even if plots failed)
    def _serialize(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    sel_path = output_dir / 'selected_cases.json'
    with open(sel_path, 'w', encoding='utf-8') as f:
        json.dump(selected_with_stats, f, ensure_ascii=False, indent=2, default=_serialize)
    print(f"  ✓ Saved: {sel_path}")

    # Compact summary table
    print(f"\n  Case summary table:")
    print(f"  {'Diagnosis':>10s} {'Outc':>4s} {'PID':>6s} {'TrueT':>5s} {'PredP':>5s} "
          f"{'Nodes':>5s} {'#Edges':>6s} {'Top edge type by alpha_e (L2)':<40s}")
    rows = []
    for c in selected_with_stats:
        if c.get('vis_stats') is None or c['patient_id'] is None:
            rows.append({'case': f"{c['diagnosis']}/{c['outcome']}", 'note': 'no case'})
            continue
        et_means = c['vis_stats']['edge_type_means_l2']
        top_et = max(et_means.items(), key=lambda x: x[1])
        print(f"  {c['diagnosis']:>10s} {c['outcome']:>4s} {c['patient_id']:>6s} "
              f"{c['true_tr_label']:>5d} {c['pred_tr_prob']:>5.3f} "
              f"{c['num_nodes']:>5d} {c['vis_stats']['n_valid_edges']:>6d} "
              f"{top_et[0]} ({top_et[1]:.3f})")
        rows.append({
            'case': f"{c['diagnosis']}/{c['outcome']}",
            'patient_id': c['patient_id'],
            'true_tr': c['true_tr_label'],
            'pred_prob': c['pred_tr_prob'],
            'n_nodes': c['num_nodes'],
            'n_edges': c['vis_stats']['n_valid_edges'],
            'top_edge_type': top_et[0],
            'top_edge_alpha': top_et[1],
            'all_edge_means_l2': et_means,
            'all_edge_means_l1': c['vis_stats'].get('edge_type_means_l1', {}),
            'beta_v_top3': c['vis_stats'].get('beta_v_top3', []),
        })
    with open(output_dir / 'case_summary_table.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ case_summary_table.json saved")

    print(f"\n{'='*70}")
    print(f"  ✓ §4.8 시각화 완료")
    print(f"  출력: {output_dir}/")
    print(f"    - cases/<diagnosis>_<outcome>_<pid>.png  (9 cases — 3 NO CASE)")
    print(f"    - hyperedge_summary.png, outcome_distribution.png")
    print(f"    - selected_cases.json, case_summary_table.json")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
