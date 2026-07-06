#!/usr/bin/env python
"""
04: 호전 라벨 생성 + Multi-task Fold 분할 스크립트.

본 스크립트는 다음을 수행한다:
1. #58 데이터 파싱 (또는 sessions.pkl 로드)
2. 환자 단위 호전 라벨 생성 (≥ 1.0 임계값)
3. 라벨 통계 출력 (사용자 검증용)
4. 주 과제용 (TR-valid) 5-fold 분할
5. 보조 과제용 (4-class 진단) 5-fold 분할
6. 환자 단위 sample 생성 (입력: 첫 3회기)
7. 산출물 저장: labels.json, fold_tr.json, fold_diag.json, samples.pkl

산출물 (모두 data/processed/ 에 저장):
  patient_labels.json       — 환자별 multi-task 라벨
  fold_tr.json              — 주 과제용 5-fold 분할
  fold_diag.json            — 보조 과제용 5-fold 분할
  patient_samples.pkl       — 환자 단위 학습 샘플
  label_statistics.json     — 통계 요약

실행 예:
    python 04_generate_labels.py \\
        --sessions data/processed/sessions.pkl \\
        --output data/processed \\
        --response-threshold 1.0 \\
        --input-window 3 \\
        --n-folds 5 --seed 42
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.labels import (
    RESPONSE_THRESHOLD,
    generate_all_patient_labels,
    save_labels_json,
)
from data.parser import parse_all_sessions
from data.splits import (
    build_patient_level_samples,
    diagnosis_cohort_kfold,
    tr_cohort_kfold,
)


def main():
    parser = argparse.ArgumentParser(description="호전 라벨 생성 + Multi-task Fold")
    parser.add_argument('--sessions', type=str, default=None,
                        help='기존 sessions.pkl 파일 (없으면 --extracted에서 새로 파싱)')
    parser.add_argument('--extracted', type=str, default=None,
                        help='--sessions 없을 시 압축 해제 디렉토리 (예: data/extracted)')
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'both'])
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='출력 디렉토리 (예: data/processed)')
    parser.add_argument('--response-threshold', type=float, default=RESPONSE_THRESHOLD,
                        help=f'호전 임계값 (기본 {RESPONSE_THRESHOLD})')
    parser.add_argument('--input-window', type=int, default=3,
                        help='환자 단위 샘플의 입력 회기 수 (기본 3)')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  04: 호전 라벨 + Multi-task Fold 생성")
    print(f"{'='*70}")
    print(f"  출력 디렉토리:      {output}")
    print(f"  호전 임계값:        ≥ {args.response_threshold}")
    print(f"  입력 window:        첫 {args.input_window} 회기")
    print(f"  5-fold seed:        {args.seed}\n")

    # ===== 1. Sessions 로드 또는 파싱 =====
    sessions = []
    if args.sessions and Path(args.sessions).exists():
        print(f"[ 1/6 ] Sessions 로드: {args.sessions}")
        with open(args.sessions, 'rb') as f:
            sessions = pickle.load(f)
        print(f"  → {len(sessions):,}개 세션 로드")
    elif args.extracted:
        print(f"[ 1/6 ] 압축 해제 디렉토리에서 새로 파싱: {args.extracted}")
        extracted = Path(args.extracted).expanduser().resolve()
        splits_to_load = ['training', 'validation'] if args.split == 'both' else [args.split]
        for sp in splits_to_load:
            labeling = extracted / sp / 'labeling'
            if not labeling.exists():
                print(f"  WARN: {labeling} 없음")
                continue
            sessions.extend(parse_all_sessions(labeling, progress=True))
        print(f"  → 총 {len(sessions):,}개 세션 파싱")
    else:
        sys.exit("ERROR: --sessions 또는 --extracted 둘 중 하나는 필수")

    if not sessions:
        sys.exit("ERROR: 사용 가능한 세션 없음")

    # ===== 2. 환자별 호전 라벨 생성 =====
    print(f"\n[ 2/6 ] 환자별 호전 라벨 생성")
    patient_labels = generate_all_patient_labels(
        sessions,
        response_threshold=args.response_threshold,
        verbose=True,
    )

    # ===== 3. 라벨 저장 =====
    print(f"[ 3/6 ] 라벨 저장")
    labels_path = output / 'patient_labels.json'
    save_labels_json(patient_labels, labels_path)

    # ===== 4. 주 과제용 fold 분할 (TR-valid 환자만) =====
    print(f"\n[ 4/6 ] 주 과제 (Treatment Response) 5-fold 분할")
    tr_fold = tr_cohort_kfold(
        sessions, patient_labels,
        n_folds=args.n_folds, seed=args.seed,
    )
    tr_fold_path = output / 'fold_tr.json'
    tr_fold.to_json(tr_fold_path)
    print(f"  → fold 저장: {tr_fold_path}")
    for fi in tr_fold.folds:
        print(f"    fold {fi.fold_index}: train {len(fi.train_patients)}명 / "
              f"test {len(fi.test_patients)}명  "
              f"(test 라벨 분포: {fi.test_label_dist})")

    # ===== 5. 보조 과제용 fold 분할 (4-class 진단, 모든 환자) =====
    print(f"\n[ 5/6 ] 보조 과제 (4-class Diagnosis) 5-fold 분할")
    diag_fold = diagnosis_cohort_kfold(
        sessions, patient_labels,
        n_folds=args.n_folds, seed=args.seed,
    )
    diag_fold_path = output / 'fold_diag.json'
    diag_fold.to_json(diag_fold_path)
    print(f"  → fold 저장: {diag_fold_path}")
    for fi in diag_fold.folds:
        print(f"    fold {fi.fold_index}: train {len(fi.train_patients)}명 / "
              f"test {len(fi.test_patients)}명  "
              f"(test 진단군 분포: {fi.test_label_dist})")

    # ===== 6. 환자 단위 샘플 생성 =====
    print(f"\n[ 6/6 ] 환자 단위 학습 샘플 생성 (입력: 첫 {args.input_window} 회기)")
    patient_samples = build_patient_level_samples(
        sessions, patient_labels,
        input_window=args.input_window,
        require_tr_valid=False,  # TR-invalid 환자도 보조 과제에는 사용
    )
    samples_path = output / 'patient_samples.pkl'
    with open(samples_path, 'wb') as f:
        pickle.dump(patient_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  → 샘플 저장: {samples_path}")
    print(f"  → 총 {len(patient_samples):,}개 환자 샘플")

    n_tr_valid = sum(1 for s in patient_samples if patient_labels[s.patient_id].tr_valid)
    print(f"      TR-valid (주 과제 학습 가능): {n_tr_valid:,}명")
    print(f"      4-class diag (보조 과제 항상 가능): {len(patient_samples):,}명")

    # 통계 요약 저장
    stats = {
        'response_threshold': args.response_threshold,
        'input_window': args.input_window,
        'n_folds': args.n_folds,
        'seed': args.seed,
        'n_sessions': len(sessions),
        'n_patients': len(patient_labels),
        'n_tr_valid_patients': sum(1 for l in patient_labels.values() if l.tr_valid),
        'n_responder': sum(1 for l in patient_labels.values() if l.tr_valid and l.tr_label == 1),
        'n_stable':    sum(1 for l in patient_labels.values() if l.tr_valid and l.tr_label == 0),
        'n_patient_samples': len(patient_samples),
    }
    stats_path = output / 'label_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n  → 통계 저장: {stats_path}")

    print(f"\n{'='*70}")
    print(f"  ✓ 라벨 + Fold 생성 완료")
    print(f"{'='*70}\n")
    print(f"다음 단계: B-block #539 텍스트 정서 인코더 사전학습\n")


if __name__ == '__main__':
    main()
