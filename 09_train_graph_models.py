#!/usr/bin/env python
"""
09: C-block 2차 — B4/B5/B6/B6v2 그래프 모델 학습.

[v2 변경]
    - --models 옵션에 B6v2 추가
    - --emotion-features 인자 추가 (B6v2 필수)
    - B6v2: 569-dim 노드 feature + 5종 hyperedge (E_affective_pattern 포함)

사용법 (B6v2 단독 학습):
    python 09_train_graph_models.py \\
        --extracted data/extracted \\
        --processed data/processed \\
        --embeddings data/embeddings \\
        --emotion-features data/emotion_features.json \\
        --output results/cblock2_b6v2 \\
        --scenarios early full \\
        --models B6v2 \\
        --seeds 0 42 2026 7 1024

사용법 (B6 + B6v2 비교):
    python 09_train_graph_models.py \\
        --extracted ... --processed ... --embeddings ... \\
        --emotion-features data/emotion_features.json \\
        --output results/cblock2_compare \\
        --models B6 B6v2
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def load_data(args):
    from data.parser import parse_all_sessions
    from data.labels import load_labels_json
    from data.splits import SplitResult

    print(f"[1/3] Sessions 파싱: {args.extracted}/{args.split}/labeling")
    labeling = Path(args.extracted) / args.split / 'labeling'
    sessions = parse_all_sessions(labeling, progress=True)
    print(f"  → {len(sessions):,} 세션 로드")

    by_patient = defaultdict(list)
    for s in sessions:
        if s.patient_id and s.session_number > 0:
            by_patient[s.patient_id].append(s)
    for pid in by_patient:
        by_patient[pid].sort(key=lambda s: s.session_number)
    print(f"  → {len(by_patient):,} 환자")

    labels_path = Path(args.processed) / 'patient_labels.json'
    patient_labels = load_labels_json(labels_path)
    print(f"[2/3] 환자 라벨 로드: {len(patient_labels):,}명")

    fold_tr_path = Path(args.processed) / 'fold_tr.json'
    fold_diag_path = Path(args.processed) / 'fold_diag.json'
    fold_tr = SplitResult.from_json(fold_tr_path)
    fold_diag = SplitResult.from_json(fold_diag_path)
    print(f"[3/3] Fold: TR {fold_tr.n_patients}명, 진단 {fold_diag.n_patients}명")

    return dict(by_patient), patient_labels, fold_tr, fold_diag


def build_graphs_for_scenario(
    by_patient, patient_labels, embeddings_dir, scenario, input_window, model_type,
    emotion_features_path=None,
):
    from graph.build_graph import build_all_patient_graphs

    print(f"\n[그래프 빌드: {model_type} / {scenario}]")
    graphs = build_all_patient_graphs(
        sessions_by_patient=by_patient,
        patient_labels=patient_labels,
        embeddings_cache_dir=Path(embeddings_dir),
        model_type=model_type,
        scenario=scenario,
        input_window=input_window,
        progress=True,
        emotion_features_path=Path(emotion_features_path) if emotion_features_path else None,
    )
    n_tr_valid = sum(1 for g in graphs.values() if g.tr_valid)

    if model_type in ('B5', 'B6', 'B6v2'):
        edge_counts = [g.num_hyperedges for g in graphs.values()]
        feat_dim = next(iter(graphs.values())).node_features.shape[1] if graphs else 0
        print(f"  → {len(graphs):,}명 / TR-valid {n_tr_valid}명 / "
              f"hyperedges mean={np.mean(edge_counts):.1f}, max={max(edge_counts)} / "
              f"node_dim={feat_dim}")
    else:
        print(f"  → {len(graphs):,}명 / TR-valid {n_tr_valid}명")
    return graphs


def aggregate_and_summarize(runs_path, output_dir, models_run):
    from training.metrics import aggregate_metrics, format_mean_ci

    all_runs = []
    with open(runs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_runs.append(json.loads(line))

    grouped = defaultdict(list)
    for r in all_runs:
        grouped[(r['model_name'], r['scenario'])].append(r)

    aggregated = {}
    for (model, scenario), runs in grouped.items():
        tr_list = [r['test_metrics'] for r in runs]
        diag_list = [r['test_diag_metrics'] for r in runs]
        tr_agg = aggregate_metrics(tr_list,
                                   keys=['auroc', 'f1_pos', 'mcc', 'balanced_acc',
                                         'precision_pos', 'recall_pos', 'accuracy'])
        diag_agg = aggregate_metrics(diag_list, keys=['macro_f1', 'accuracy'])

        per_diag_runs = defaultdict(list)
        for r in runs:
            for dn, dtr in r.get('test_per_diag_tr', {}).items():
                if dtr is not None:
                    per_diag_runs[dn].append(dtr)
        per_diag_agg = {dn: aggregate_metrics(sub, keys=['auroc', 'f1_pos', 'mcc'])
                        for dn, sub in per_diag_runs.items()}

        aggregated[f"{model}_{scenario}"] = {
            'n_runs': len(runs),
            'tr_metrics': tr_agg,
            'diag_metrics': diag_agg,
            'per_diag_tr': per_diag_agg,
        }

    agg_path = output_dir / 'cblock2_aggregated.json'
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"  → 집계 저장: {agg_path}")

    # 표 출력 (실제 학습한 모델만)
    lines = [f"# C-block 2차 (B6v2) 결과 — 학습 모델: {', '.join(models_run)}\n"]

    for scenario in ['early', 'full']:
        scen_label = "Early Prediction (첫 3회기)" if scenario == 'early' else "Full Scenario (전 가용 회기)"
        lines.append(f"\n## 표 4.1{'a' if scenario == 'early' else 'b'}_v2 — {scen_label} (TR)\n")
        lines.append("| 모델 | AUROC ↑ | F1(pos) ↑ | MCC ↑ | Balanced Acc ↑ |")
        lines.append("|---|---|---|---|---|")
        for model in models_run:
            key = f"{model}_{scenario}"
            if key not in aggregated:
                lines.append(f"| {model} | — | — | — | — |")
                continue
            tr = aggregated[key]['tr_metrics']
            row = [model,
                   format_mean_ci(tr.get('auroc', {})),
                   format_mean_ci(tr.get('f1_pos', {})),
                   format_mean_ci(tr.get('mcc', {})),
                   format_mean_ci(tr.get('balanced_acc', {}))]
            lines.append("| " + " | ".join(row) + " |")

        lines.append(f"\n### 보조 task: 4-class 진단 ({scen_label})\n")
        lines.append("| 모델 | Macro F1 ↑ | Accuracy ↑ |")
        lines.append("|---|---|---|")
        for model in models_run:
            key = f"{model}_{scenario}"
            if key not in aggregated:
                lines.append(f"| {model} | — | — |")
                continue
            d = aggregated[key]['diag_metrics']
            lines.append(f"| {model} | {format_mean_ci(d.get('macro_f1', {}))} | "
                         f"{format_mean_ci(d.get('accuracy', {}))} |")

    # 표 4.2 진단군별 (B6v2 또는 B6 중 학습된 것 우선)
    for tgt_model in ['B6v2', 'B6']:
        key = f'{tgt_model}_early'
        if key in aggregated:
            lines.append(f"\n\n## 표 4.2 — 진단군별 TR 분해 ({tgt_model}, Early)\n")
            lines.append("| 진단군 | AUROC ↑ | F1(pos) ↑ | MCC ↑ |")
            lines.append("|---|---|---|---|")
            for dn in ['depression', 'anxiety', 'addiction']:
                pd = aggregated[key].get('per_diag_tr', {}).get(dn, {})
                row = [dn,
                       format_mean_ci(pd.get('auroc', {})),
                       format_mean_ci(pd.get('f1_pos', {})),
                       format_mean_ci(pd.get('mcc', {}))]
                lines.append("| " + " | ".join(row) + " |")
            break

    summary_path = output_dir / 'cblock2_summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  → 표 형식 요약 저장: {summary_path}")
    print("\n" + '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="C-block 2차 그래프 모델 학습 (B6v2 지원)")
    parser.add_argument('--extracted', required=True, type=str)
    parser.add_argument('--split', default='training', choices=['training', 'validation'])
    parser.add_argument('--processed', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--emotion-features', type=str, default=None,
                       help='B6v2 학습 시 필수: data/emotion_features.json 경로')
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--scenarios', nargs='+', default=['early', 'full'],
                        choices=['early', 'full'])
    parser.add_argument('--models', nargs='+', default=['B6v2'],
                        choices=['B4', 'B5', 'B6', 'B6v2'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 42, 2026, 7, 1024])
    parser.add_argument('--input-window', type=int, default=3)

    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lambda-tr', type=float, default=1.0)
    parser.add_argument('--lambda-diag', type=float, default=0.3)
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    args = parser.parse_args()

    # B6v2가 모델 리스트에 있으면 emotion-features 필수
    if 'B6v2' in args.models and args.emotion_features is None:
        sys.exit("ERROR: B6v2 학습 시 --emotion-features 인자 필수.\n"
                 "       먼저 scripts/extract_emotion_features.py 로 emotion_features.json 생성.")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'batch_size': args.batch_size, 'max_epochs': args.max_epochs,
        'patience': args.patience,
        'lambda_tr': args.lambda_tr, 'lambda_diag': args.lambda_diag,
        'focal_alpha': args.focal_alpha, 'focal_gamma': args.focal_gamma,
    }
    with open(output_dir / 'cblock2_config.json', 'w', encoding='utf-8') as f:
        json.dump({**config, 'scenarios': args.scenarios, 'models': args.models,
                  'seeds': args.seeds, 'input_window': args.input_window,
                  'emotion_features': args.emotion_features},
                  f, ensure_ascii=False, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  C-block 2차 (B6v2) — 그래프 모델 학습")
    print(f"{'='*70}")
    print(f"  Device:    {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU:       {p.name}, {p.total_memory / 1e9:.1f} GB")
    print(f"  Models:    {args.models}")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Seeds:     {args.seeds}")
    if args.emotion_features:
        print(f"  Emotion:   {args.emotion_features}")
    total = len(args.models) * len(args.scenarios) * 5 * len(args.seeds)
    print(f"  학습 횟수: {len(args.models)} × {len(args.scenarios)} × 5 fold × "
          f"{len(args.seeds)} seed = {total}회\n")

    by_patient, patient_labels, fold_tr, fold_diag = load_data(args)

    all_runs_files = []
    for model_name in args.models:
        for scenario in args.scenarios:
            graphs = build_graphs_for_scenario(
                by_patient, patient_labels, args.embeddings,
                scenario, args.input_window, model_name,
                emotion_features_path=args.emotion_features,
            )

            runs_path = output_dir / f'cblock2_runs_{model_name}_{scenario}.jsonl'
            runs_file = open(runs_path, 'w', encoding='utf-8')

            from training.graph_trainer import (
                train_single_run, compute_diagnosis_class_weights,
            )

            print(f"\n{'='*70}")
            print(f"  {model_name} ({scenario}) — {fold_tr.n_folds}-fold × {len(args.seeds)}-seed")
            print(f"{'='*70}")

            run_idx = 0
            start_all = time.time()
            total_runs = fold_tr.n_folds * len(args.seeds)

            for fold_info in fold_tr.folds:
                train_pids = set(fold_info.train_patients)
                test_pids = set(fold_info.test_patients)

                train_graphs = sorted(
                    [g for pid, g in graphs.items() if pid in train_pids],
                    key=lambda g: g.patient_id,
                )
                test_graphs = [g for pid, g in graphs.items() if pid in test_pids]

                n_val = max(1, int(len(train_graphs) * 0.1))
                val_graphs = train_graphs[:n_val]
                train_graphs_actual = train_graphs[n_val:]

                diag_weights = compute_diagnosis_class_weights(train_graphs_actual)

                for seed in args.seeds:
                    run_idx += 1
                    start = time.time()
                    result = train_single_run(
                        model_name=model_name,
                        train_graphs=train_graphs_actual,
                        val_graphs=val_graphs,
                        test_graphs=test_graphs,
                        diagnosis_class_weights=diag_weights,
                        config=config, device=device,
                        scenario=scenario, fold=fold_info.fold_index, seed=seed,
                        verbose=False,
                    )
                    elapsed = time.time() - start
                    runs_file.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
                    runs_file.flush()

                    total_elapsed = time.time() - start_all
                    avg = total_elapsed / run_idx
                    remaining = (total_runs - run_idx) * avg
                    print(f"  [{run_idx}/{total_runs}] {model_name} f={fold_info.fold_index} "
                          f"s={seed}  auroc={result.test_metrics['auroc']:.3f} "
                          f"f1={result.test_metrics['f1_pos']:.3f} "
                          f"mcc={result.test_metrics['mcc']:.3f} "
                          f"diag_f1={result.test_diag_metrics['macro_f1']:.3f} "
                          f"({elapsed:.1f}s, ~{remaining/60:.1f}분)")

            runs_file.close()
            all_runs_files.append(runs_path)
            print(f"\n  ✓ {model_name} ({scenario}) 완료: {time.time() - start_all:.0f}초")

            del graphs
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    combined_path = output_dir / 'cblock2_runs.jsonl'
    with open(combined_path, 'w', encoding='utf-8') as fout:
        for rf in all_runs_files:
            with open(rf, 'r', encoding='utf-8') as fin:
                fout.write(fin.read())

    print(f"\n{'='*70}")
    print(f"  결과 집계 및 표 형식 출력")
    print(f"{'='*70}\n")
    aggregate_and_summarize(combined_path, output_dir, args.models)

    print(f"\n{'='*70}")
    print(f"  ✓ C-block 2차 (B6v2) 완료")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
