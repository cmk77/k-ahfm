#!/usr/bin/env python
"""
08: C-block 1차 — B1/B2/B3 baseline 학습 (5-fold × 5-seed = 25회 반복).

사용법:
    python 08_train_baselines.py \\
        --extracted <repo>/data/extracted \\
        --processed <repo>/data/processed \\
        --embeddings <repo>/data/embeddings \\
        --output <repo>/results/cblock1 \\
        --scenarios early full \\
        --models B1 B2 B3 \\
        --seeds 0 42 2026 7 1024

학습 구조:
    각 model × scenario × fold × seed = 1회 학습.
    총 학습 횟수: 3 (B1,B2,B3) × 2 (early,full) × 5 (folds) × 5 (seeds) = 150회.
    각 학습 ~ 30-60초 (paragraph 임베딩 캐싱 덕분에) → 총 1.5-2.5시간.

산출물:
    {output}/cblock1_runs.jsonl          — 모든 (model,scenario,fold,seed) 결과
    {output}/cblock1_aggregated.json     — 25회 반복 평균 ± 95% CI
    {output}/cblock1_summary.md          — 표 4.1a/4.1b 셀에 채울 형식
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# 본 스크립트는 K-AHFM 프로젝트 루트(<repo>)에서 실행한다고 가정한다.
# 다음 sys.path 설정으로 기존 src/data 모듈 + 본 패키지 src 모두 접근 가능.
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def load_data(args):
    """필요한 데이터를 모두 로드."""
    from data.parser import parse_all_sessions
    from data.labels import load_labels_json, PatientLabel

    # 1. Sessions 파싱
    print(f"[1/3] Sessions 파싱: {args.extracted}/{args.split}/labeling")
    labeling = Path(args.extracted) / args.split / 'labeling'
    sessions = parse_all_sessions(labeling, progress=True)
    print(f"  → {len(sessions):,} 세션 로드")

    # 환자별 세션 그룹화 (시간순)
    by_patient = defaultdict(list)
    for s in sessions:
        if s.patient_id and s.session_number > 0:
            by_patient[s.patient_id].append(s)
    for pid in by_patient:
        by_patient[pid].sort(key=lambda s: s.session_number)
    print(f"  → {len(by_patient):,} 환자")

    # 2. 환자 라벨 로드
    labels_path = Path(args.processed) / 'patient_labels.json'
    patient_labels = load_labels_json(labels_path)
    print(f"[2/3] 환자 라벨 로드: {len(patient_labels):,}명")

    # 3. fold 로드
    from data.splits import SplitResult
    fold_tr_path = Path(args.processed) / 'fold_tr.json'
    fold_diag_path = Path(args.processed) / 'fold_diag.json'
    fold_tr = SplitResult.from_json(fold_tr_path)
    fold_diag = SplitResult.from_json(fold_diag_path)
    print(f"[3/3] Fold 로드: TR {fold_tr.n_patients}명, 진단 {fold_diag.n_patients}명")

    return by_patient, patient_labels, fold_tr, fold_diag


def extract_features_for_scenario(
    by_patient, patient_labels, embeddings_dir, scenario, input_window,
):
    """주어진 scenario(early/full)의 환자 feature를 모두 추출."""
    from utils.features import extract_all_patient_features

    print(f"\n[Feature 추출: {scenario}]")
    features = extract_all_patient_features(
        sessions_by_patient=dict(by_patient),
        patient_labels=patient_labels,
        embeddings_cache_dir=Path(embeddings_dir),
        scenario=scenario,
        input_window=input_window,
        progress=True,
    )
    n_tr_valid = sum(1 for f in features.values() if f.tr_valid)
    print(f"  → {len(features):,}명 feature 추출 (TR-valid: {n_tr_valid}명)")
    return features


def run_all_experiments(
    features, fold_tr, fold_diag, model_names, seeds, scenario, device, config, output_dir,
):
    """모든 (model, fold, seed) 조합 학습."""
    from models.baselines import get_feature_key
    from training.trainer import train_single_run, compute_diagnosis_class_weights

    # TR fold 사용 (학습 코호트가 다르므로). 진단 보조 task는 같은 fold 내 환자에서 동시 학습.
    # 즉 TR-valid 환자만 학습 코호트로 사용하며, 보조 task의 진단 라벨은 모든 환자에서 활용.
    # 단, 본 baseline에서는 TR fold 환자만 사용 (보조 task는 TR-valid 코호트 내에서 학습).

    runs_path = output_dir / 'cblock1_runs.jsonl'
    runs_file = open(runs_path, 'w', encoding='utf-8')

    total_runs = len(model_names) * fold_tr.n_folds * len(seeds)
    run_idx = 0
    start_all = time.time()

    for model_name in model_names:
        feature_key = get_feature_key(model_name)
        print(f"\n{'='*70}")
        print(f"  {model_name} ({scenario}) — {fold_tr.n_folds}-fold × {len(seeds)}-seed = "
              f"{fold_tr.n_folds * len(seeds)}회")
        print(f"{'='*70}")

        for fold_info in fold_tr.folds:
            train_pids = set(fold_info.train_patients)
            test_pids = set(fold_info.test_patients)

            # train/test feature 분리
            train_features = [f for pid, f in features.items() if pid in train_pids]
            test_features = [f for pid, f in features.items() if pid in test_pids]

            # train 내에서 validation split (10%)
            n_val = max(1, int(len(train_features) * 0.1))
            # 환자 ID 정렬해서 deterministic
            train_features.sort(key=lambda f: f.patient_id)
            val_features = train_features[:n_val]
            train_features_actual = train_features[n_val:]

            # 진단 클래스 가중치 (train 기준)
            diag_weights = compute_diagnosis_class_weights(train_features_actual)

            for seed in seeds:
                run_idx += 1
                start = time.time()
                result = train_single_run(
                    model_name=model_name,
                    train_features=train_features_actual,
                    val_features=val_features,
                    test_features=test_features,
                    feature_key=feature_key,
                    diagnosis_class_weights=diag_weights,
                    config=config,
                    device=device,
                    scenario=scenario,
                    fold=fold_info.fold_index,
                    seed=seed,
                    verbose=False,
                )
                elapsed = time.time() - start

                # JSONL 저장
                result_dict = asdict(result)
                runs_file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                runs_file.flush()

                # 진행 상황
                total_elapsed = time.time() - start_all
                avg_per_run = total_elapsed / run_idx
                remaining = (total_runs - run_idx) * avg_per_run
                print(f"  [{run_idx}/{total_runs}] {model_name} fold={fold_info.fold_index} "
                      f"seed={seed}  auroc={result.test_metrics['auroc']:.3f}  "
                      f"f1_pos={result.test_metrics['f1_pos']:.3f}  "
                      f"mcc={result.test_metrics['mcc']:.3f}  "
                      f"diag_f1={result.test_diag_metrics['macro_f1']:.3f}  "
                      f"({elapsed:.1f}s, 남은시간 ~{remaining/60:.1f}분)")

    runs_file.close()
    print(f"\n  ✓ 모든 학습 완료: {time.time() - start_all:.0f}초 ({(time.time()-start_all)/60:.1f}분)")
    print(f"  → 결과 저장: {runs_path}")
    return runs_path


def aggregate_and_summarize(runs_path, output_dir):
    """JSONL 결과를 집계하여 표 4.1 형식으로 출력."""
    from training.metrics import aggregate_metrics, format_mean_ci

    # 로드
    all_runs = []
    with open(runs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_runs.append(json.loads(line))

    # (model, scenario)별 집계
    grouped = defaultdict(list)
    for r in all_runs:
        key = (r['model_name'], r['scenario'])
        grouped[key].append(r)

    aggregated = {}
    for (model, scenario), runs in grouped.items():
        tr_metrics_list = [r['test_metrics'] for r in runs]
        diag_metrics_list = [r['test_diag_metrics'] for r in runs]
        tr_agg = aggregate_metrics(tr_metrics_list,
                                   keys=['auroc', 'f1_pos', 'mcc', 'balanced_acc',
                                         'precision_pos', 'recall_pos', 'accuracy'])
        diag_agg = aggregate_metrics(diag_metrics_list,
                                     keys=['macro_f1', 'accuracy'])

        # 진단군별 TR (4.2 표용)
        per_diag_runs = defaultdict(list)
        for r in runs:
            for diag_name, diag_tr in r.get('test_per_diag_tr', {}).items():
                if diag_tr is not None:
                    per_diag_runs[diag_name].append(diag_tr)
        per_diag_agg = {}
        for diag_name, sub_runs in per_diag_runs.items():
            per_diag_agg[diag_name] = aggregate_metrics(
                sub_runs, keys=['auroc', 'f1_pos', 'mcc'],
            )

        aggregated[f"{model}_{scenario}"] = {
            'n_runs': len(runs),
            'tr_metrics': tr_agg,
            'diag_metrics': diag_agg,
            'per_diag_tr': per_diag_agg,
        }

    # 저장
    agg_path = output_dir / 'cblock1_aggregated.json'
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"  → 집계 저장: {agg_path}")

    # 표 4.1a/4.1b 형식 마크다운 출력
    summary_lines = []
    summary_lines.append("# C-block 1차 결과 — 표 4.1 (B1-B3 baseline)\n")

    for scenario in ['early', 'full']:
        scen_label = "Early Prediction (첫 3회기)" if scenario == 'early' else "Full Scenario (전 가용 회기)"
        summary_lines.append(f"\n## 표 4.1{'a' if scenario == 'early' else 'b'} — {scen_label}\n")
        summary_lines.append("| 모델 | AUROC ↑ | F1(pos) ↑ | MCC ↑ | Balanced Acc ↑ |")
        summary_lines.append("|---|---|---|---|---|")
        for model in ['B1', 'B2', 'B3']:
            key = f"{model}_{scenario}"
            if key not in aggregated:
                summary_lines.append(f"| {model} | — | — | — | — |")
                continue
            tr = aggregated[key]['tr_metrics']
            row = [
                model,
                format_mean_ci(tr.get('auroc', {})),
                format_mean_ci(tr.get('f1_pos', {})),
                format_mean_ci(tr.get('mcc', {})),
                format_mean_ci(tr.get('balanced_acc', {})),
            ]
            summary_lines.append("| " + " | ".join(row) + " |")

        summary_lines.append("\n### 보조 task: 4-class 진단\n")
        summary_lines.append("| 모델 | Macro F1 ↑ | Accuracy ↑ |")
        summary_lines.append("|---|---|---|")
        for model in ['B1', 'B2', 'B3']:
            key = f"{model}_{scenario}"
            if key not in aggregated:
                summary_lines.append(f"| {model} | — | — |")
                continue
            d = aggregated[key]['diag_metrics']
            summary_lines.append(
                f"| {model} | {format_mean_ci(d.get('macro_f1', {}))} | "
                f"{format_mean_ci(d.get('accuracy', {}))} |"
            )

    summary_path = output_dir / 'cblock1_summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"  → 표 형식 요약 저장: {summary_path}")

    # 콘솔 출력
    print("\n" + '\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description="C-block 1차 baseline 학습")
    parser.add_argument('--extracted', required=True, type=str,
                        help='압축 해제 디렉토리 (예: <repo>/data/extracted)')
    parser.add_argument('--split', default='training', choices=['training', 'validation'])
    parser.add_argument('--processed', required=True, type=str,
                        help='라벨/fold 디렉토리 (예: <repo>/data/processed)')
    parser.add_argument('--embeddings', required=True, type=str,
                        help='paragraph 임베딩 캐시 (예: <repo>/data/embeddings)')
    parser.add_argument('--output', required=True, type=str,
                        help='결과 출력 디렉토리 (예: <repo>/results/cblock1)')
    parser.add_argument('--scenarios', nargs='+', default=['early', 'full'],
                        choices=['early', 'full'])
    parser.add_argument('--models', nargs='+', default=['B1', 'B2', 'B3'],
                        choices=['B1', 'B2', 'B3'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 42, 2026, 7, 1024])
    parser.add_argument('--input-window', type=int, default=3,
                        help='early scenario 회기 수 (기본 3)')

    # 학습 하이퍼파라미터
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lambda-tr', type=float, default=1.0)
    parser.add_argument('--lambda-diag', type=float, default=0.3)
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'lambda_tr': args.lambda_tr,
        'lambda_diag': args.lambda_diag,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
    }
    with open(output_dir / 'cblock1_config.json', 'w', encoding='utf-8') as f:
        json.dump({**config, 'scenarios': args.scenarios, 'models': args.models,
                  'seeds': args.seeds, 'input_window': args.input_window},
                  f, ensure_ascii=False, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  C-block 1차 — B1/B2/B3 baseline 학습")
    print(f"{'='*70}")
    print(f"  Device:          {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU:             {p.name}")
    print(f"  Models:          {args.models}")
    print(f"  Scenarios:       {args.scenarios}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  학습 횟수:       {len(args.models)} × {len(args.scenarios)} × 5 fold × "
          f"{len(args.seeds)} seed = "
          f"{len(args.models) * len(args.scenarios) * 5 * len(args.seeds)}회\n")

    # 데이터 로드 (1회)
    by_patient, patient_labels, fold_tr, fold_diag = load_data(args)

    # 각 scenario별로 feature 추출 → 학습
    all_runs_files = []
    for scenario in args.scenarios:
        features = extract_features_for_scenario(
            by_patient, patient_labels, args.embeddings,
            scenario, args.input_window,
        )

        # 학습
        runs_path = output_dir / f'cblock1_runs_{scenario}.jsonl'
        # 실행
        runs_file = open(runs_path, 'w', encoding='utf-8')

        total_runs = len(args.models) * fold_tr.n_folds * len(args.seeds)
        run_idx = 0
        start_all = time.time()

        from models.baselines import get_feature_key
        from training.trainer import train_single_run, compute_diagnosis_class_weights

        for model_name in args.models:
            feature_key = get_feature_key(model_name)
            print(f"\n{'='*70}")
            print(f"  {model_name} ({scenario}) — {fold_tr.n_folds}-fold × {len(args.seeds)}-seed = "
                  f"{fold_tr.n_folds * len(args.seeds)}회")
            print(f"{'='*70}")

            for fold_info in fold_tr.folds:
                train_pids = set(fold_info.train_patients)
                test_pids = set(fold_info.test_patients)

                train_features = sorted(
                    [f for pid, f in features.items() if pid in train_pids],
                    key=lambda f: f.patient_id,
                )
                test_features = [f for pid, f in features.items() if pid in test_pids]

                # validation split
                n_val = max(1, int(len(train_features) * 0.1))
                val_features = train_features[:n_val]
                train_features_actual = train_features[n_val:]

                diag_weights = compute_diagnosis_class_weights(train_features_actual)

                for seed in args.seeds:
                    run_idx += 1
                    start = time.time()
                    result = train_single_run(
                        model_name=model_name,
                        train_features=train_features_actual,
                        val_features=val_features,
                        test_features=test_features,
                        feature_key=feature_key,
                        diagnosis_class_weights=diag_weights,
                        config=config,
                        device=device,
                        scenario=scenario,
                        fold=fold_info.fold_index,
                        seed=seed,
                        verbose=False,
                    )
                    elapsed = time.time() - start
                    runs_file.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
                    runs_file.flush()

                    total_elapsed = time.time() - start_all
                    avg_per_run = total_elapsed / run_idx
                    remaining = (total_runs - run_idx) * avg_per_run
                    print(f"  [{run_idx}/{total_runs}] {model_name} f={fold_info.fold_index} "
                          f"s={seed}  auroc={result.test_metrics['auroc']:.3f} "
                          f"f1={result.test_metrics['f1_pos']:.3f} "
                          f"mcc={result.test_metrics['mcc']:.3f} "
                          f"diag_f1={result.test_diag_metrics['macro_f1']:.3f} "
                          f"({elapsed:.1f}s, ~{remaining/60:.1f}분 남음)")

        runs_file.close()
        all_runs_files.append(runs_path)
        print(f"\n  ✓ {scenario} 완료: {time.time() - start_all:.0f}초")

    # 모든 scenario 결과 통합 JSONL
    combined_path = output_dir / 'cblock1_runs.jsonl'
    with open(combined_path, 'w', encoding='utf-8') as fout:
        for rf in all_runs_files:
            with open(rf, 'r', encoding='utf-8') as fin:
                fout.write(fin.read())

    # 집계 및 표 4.1 형식 출력
    print(f"\n{'='*70}")
    print(f"  결과 집계 및 표 형식 출력")
    print(f"{'='*70}\n")
    aggregate_and_summarize(combined_path, output_dir)

    print(f"\n{'='*70}")
    print(f"  ✓ C-block 1차 완료")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
