#!/usr/bin/env python
"""
A1-A5 통합 실행 스크립트.

A1: JSON paragraph 파싱 → Session 객체
A2: 28 증상요인 정규화 → 세션별 특성 벡터
A3: Prosodic proxy 추출 → paragraph 단위 / 세션 단위 메타
A4: 환자 단위 5-fold 분할 → fold_indices.json
A5: PyTorch Dataset 인스턴스 생성 검증

산출물:
  data/processed/
    sessions.pkl        : Session 객체 리스트 (pickle)
    fold_indices.json   : 5-fold 분할 정보
    longitudinal_samples.pkl : sliding window 종단 샘플
    statistics.json     : 데이터 통계 요약

실행 예:
    python 03_build_dataset.py \\
        --extracted ~/projects/k-ahfm-clinic/data/extracted \\
        --output ~/projects/k-ahfm-clinic/data/processed \\
        --n-folds 5 --seed 42
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.parser import parse_all_sessions
from data.splits import (
    build_longitudinal_samples,
    fold_train_test_samples,
    patient_disjoint_kfold,
)
from data.dataset import K58LongitudinalDataset, K58SessionDataset


def main():
    parser = argparse.ArgumentParser(description="A1-A5 데이터 파이프라인")
    parser.add_argument('--extracted', '-e', required=True, type=str,
                        help='압축 해제 루트 (예: data/extracted)')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='가공 데이터 저장 디렉토리 (예: data/processed)')
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'both'],
                        help='어느 split을 사용할지 (기본: training)')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-observed', type=int, default=1,
                        help='종단 샘플 최소 관측 회기 수')
    parser.add_argument('--max-observed', type=int, default=None,
                        help='종단 샘플 최대 관측 회기 수 (None=무제한)')
    args = parser.parse_args()

    extracted = Path(args.extracted).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  A1-A5 데이터 파이프라인")
    print(f"{'='*70}")
    print(f"  추출 디렉토리:   {extracted}")
    print(f"  출력 디렉토리:   {output}")
    print(f"  5-fold seed:    {args.seed}")
    print(f"  종단 샘플 옵션:  min_obs={args.min_observed}, max_obs={args.max_observed}\n")

    # ===== A1: JSON 파싱 =====
    print("\n[ A1 ] JSON paragraph 파싱")
    splits_to_load = ['training', 'validation'] if args.split == 'both' else [args.split]
    all_sessions = []
    for sp in splits_to_load:
        labeling = extracted / sp / 'labeling'
        if not labeling.exists():
            print(f"  WARN: {labeling} 없음 — 건너뜀")
            continue
        sessions = parse_all_sessions(labeling, progress=True)
        all_sessions.extend(sessions)
    print(f"  → 총 {len(all_sessions):,}개 세션 파싱")

    if not all_sessions:
        print("ERROR: 파싱된 세션이 없음. 압축 해제가 정상적으로 되었는지 확인하세요.")
        sys.exit(1)

    # ===== A2 + A3: 정규화 + prosodic은 Dataset 내부에서 lazy 적용됨 =====
    print("\n[ A2/A3 ] 정규화 및 prosodic proxy는 Dataset 내부에서 lazy 적용")

    # 세션 단위 sanity check
    from data.normalize import session_symptoms_28
    from data.prosodic import compute_session_prosodic_stats
    s_test = all_sessions[0]
    sym = session_symptoms_28(s_test)
    pros = compute_session_prosodic_stats(s_test)
    print(f"  sanity check: 첫 세션 {s_test.patient_id} s{s_test.session_number:02d}")
    print(f"    28-증상 벡터 shape: {sym.shape}, mean: {sym.mean():.4f}, max: {sym.max():.4f}")
    print(f"    prosodic stats: silence_ratio={pros.silence_ratio:.3f}, "
          f"client_speech_ratio={pros.client_speech_ratio:.3f}, "
          f"num_turns_norm={pros.num_turns_norm:.3f}")

    # ===== A4: 환자 단위 5-fold 분할 =====
    print(f"\n[ A4 ] 환자 단위 {args.n_folds}-fold 분할 (seed={args.seed})")
    split_result = patient_disjoint_kfold(all_sessions, n_folds=args.n_folds, seed=args.seed)
    fold_path = output / 'fold_indices.json'
    split_result.to_json(fold_path)
    print(f"  → fold 정보 저장: {fold_path}")
    for fi in split_result.folds:
        print(f"    fold {fi.fold_index}: train {len(fi.train_patients)}명 "
              f"({fi.train_session_count}세션) | "
              f"test {len(fi.test_patients)}명 ({fi.test_session_count}세션)")

    # ===== A5: 종단 샘플 생성 + Dataset 인스턴스 검증 =====
    print(f"\n[ A5 ] sliding window 종단 샘플 생성")
    samples = build_longitudinal_samples(
        all_sessions,
        min_observed=args.min_observed,
        max_observed=args.max_observed,
    )
    print(f"  → 총 {len(samples):,}개 종단 학습 샘플")

    # 진단군별 샘플 분포
    sample_diag = Counter(s.diagnosis for s in samples)
    print(f"  진단군별 종단 샘플 분포:")
    for diag, n in sorted(sample_diag.items()):
        print(f"    {diag:12s}: {n:5d}")

    # 관측 길이 분포
    obs_lens = [len(s.observed_session_numbers) for s in samples]
    print(f"  관측 시퀀스 길이 분포:")
    print(f"    mean={sum(obs_lens)/len(obs_lens):.2f}, "
          f"min={min(obs_lens)}, max={max(obs_lens)}")
    len_counter = Counter(obs_lens)
    for l in sorted(len_counter.keys()):
        print(f"    길이 {l}: {len_counter[l]:4d}개 샘플")

    # Dataset 인스턴스 sanity check
    print(f"\n  Dataset 인스턴스 sanity check:")
    full_ds = K58LongitudinalDataset(all_sessions, samples)
    item = full_ds[0]
    print(f"    [0]: patient={item['patient_id']}, "
          f"obs_sessions={item['observed_session_numbers']}, "
          f"target_session={item['target_session_number']}")
    print(f"    observed_symptoms shape: {item['observed_symptoms'].shape}")
    print(f"    target_symptoms shape: {item['target_symptoms'].shape}")
    print(f"    target_symptoms (28): mean={item['target_symptoms'].mean():.4f}")

    # ===== 직렬화 저장 =====
    print(f"\n[ Save ] 결과 저장")
    sessions_path = output / 'sessions.pkl'
    with open(sessions_path, 'wb') as f:
        pickle.dump(all_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  → {sessions_path} ({sessions_path.stat().st_size / 1e6:.1f} MB)")

    samples_path = output / 'longitudinal_samples.pkl'
    with open(samples_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  → {samples_path} ({samples_path.stat().st_size / 1e6:.1f} MB)")

    # 통계 요약
    stats = {
        'n_sessions': len(all_sessions),
        'n_patients': split_result.n_patients,
        'n_folds': args.n_folds,
        'seed': args.seed,
        'n_longitudinal_samples': len(samples),
        'diagnosis_session_count': dict(Counter(s.diagnosis for s in all_sessions)),
        'diagnosis_patient_count': dict(Counter(split_result.patient_to_diagnosis.values())),
        'diagnosis_sample_count': dict(sample_diag),
        'observed_length_distribution': dict(len_counter),
    }
    stats_path = output / 'statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  → {stats_path}")

    print(f"\n{'='*70}")
    print(f"  파이프라인 완료")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
