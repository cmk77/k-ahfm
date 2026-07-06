#!/usr/bin/env python
"""
압축 해제된 #58 데이터 무결성 및 통계 검증 스크립트.

목적:
- JSON 파일이 모두 정상 파싱 가능한가
- 진단군별 / 회기별 실제 환자·세션 수 확인 (사용자 제공 정보 50/46/48 검증)
- 28 증상요인 점수의 실제 분포 확인
- 같은 환자가 여러 회기에 걸쳐 존재하는가 (종단 데이터 검증)

실행 예:
    python 02_verify_data.py --extracted ~/projects/k-ahfm-clinic/data/extracted
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from data.parser import parse_all_sessions
from data.normalize import session_symptoms_28
from data.constants import SYMPTOM_KEYS


def verify(extracted_dir: Path, split: str = 'training'):
    """검증 메인."""
    labeling_dir = extracted_dir / split / 'labeling'
    if not labeling_dir.exists():
        print(f"ERROR: 라벨링 디렉토리 없음: {labeling_dir}")
        return None

    print(f"\n{'='*70}")
    print(f"#58 데이터 검증: {labeling_dir}")
    print(f"{'='*70}\n")

    sessions = parse_all_sessions(labeling_dir, progress=True)
    if not sessions:
        print("ERROR: 파싱된 세션이 없음")
        return None

    # ----- 진단군별 세션 분포 -----
    print(f"\n[1] 진단군별 세션 수")
    diag_session = Counter(s.diagnosis for s in sessions)
    for diag, n in sorted(diag_session.items()):
        print(f"    {diag:12s}: {n:5d}개 세션")
    print(f"    {'합계':12s}: {sum(diag_session.values()):5d}개")

    # ----- 진단군 × 회기별 분포 (사용자 제공 50/46/48 검증) -----
    print(f"\n[2] 진단군 × 회기별 세션 수")
    diag_session_count: defaultdict = defaultdict(Counter)
    for s in sessions:
        diag_session_count[s.diagnosis][s.session_number] += 1

    max_session = max((max(c.keys()) for c in diag_session_count.values() if c), default=0)
    header = f"    {'diagnosis':<12s} " + " ".join(f"{n:>4d}회" for n in range(1, max_session + 1)) + "  total"
    print(header)
    print(f"    {'-'*12} " + " ".join("-----" for _ in range(max_session)) + "  -----")
    for diag in sorted(diag_session_count.keys()):
        counts = diag_session_count[diag]
        row = f"    {diag:<12s} " + " ".join(f"{counts.get(n, 0):>5d}" for n in range(1, max_session + 1))
        row += f"  {sum(counts.values()):>5d}"
        print(row)

    # ----- 환자 수 / 환자당 세션 수 -----
    print(f"\n[3] 환자별 통계")
    patient_to_diag = {s.patient_id: s.diagnosis for s in sessions if s.patient_id}
    patient_session_count: defaultdict = defaultdict(int)
    for s in sessions:
        if s.patient_id:
            patient_session_count[s.patient_id] += 1

    diag_patient = Counter(patient_to_diag.values())
    print(f"    진단군별 환자 수:")
    for diag, n in sorted(diag_patient.items()):
        print(f"      {diag:12s}: {n:4d}명")
    print(f"    환자당 세션 수 분포:")
    sc = list(patient_session_count.values())
    print(f"      mean={np.mean(sc):.2f}, median={np.median(sc):.0f}, "
          f"min={min(sc)}, max={max(sc)}")
    sc_counter = Counter(sc)
    for n_sessions in sorted(sc_counter.keys()):
        print(f"      {n_sessions:2d}회기 진행: {sc_counter[n_sessions]:4d}명")

    # ----- 28 증상요인 분포 -----
    print(f"\n[4] 28 증상요인 점수 분포 (세션 평균 집계)")
    symptom_matrix = np.stack([session_symptoms_28(s) for s in sessions])  # (N_sessions, 28)
    print(f"    행렬 크기: {symptom_matrix.shape}")
    print(f"    전체 평균: {symptom_matrix.mean():.4f}")
    print(f"    전체 표준편차: {symptom_matrix.std():.4f}")
    print(f"    0인 셀의 비율: {(symptom_matrix == 0).mean():.4f}")
    print(f"\n    증상별 평균 활성도 (상위 10):")
    means = symptom_matrix.mean(axis=0)
    top_idx = np.argsort(means)[::-1][:10]
    for i in top_idx:
        print(f"      {SYMPTOM_KEYS[i]:30s}: {means[i]:.4f}")

    # ----- 회기 간 증상 변화 (종단 신호 확인) -----
    print(f"\n[5] 종단 신호: 같은 환자의 회기 간 증상 변화")
    # 환자별로 회기 정렬 후, 연속 회기 간 증상 차이의 평균 크기를 측정
    patient_sessions: defaultdict = defaultdict(list)
    for s in sessions:
        if s.patient_id:
            patient_sessions[s.patient_id].append(s)
    for pid in patient_sessions:
        patient_sessions[pid].sort(key=lambda x: x.session_number)

    delta_norms = []
    for pid, ss_list in patient_sessions.items():
        if len(ss_list) < 2:
            continue
        for i in range(len(ss_list) - 1):
            s1, s2 = session_symptoms_28(ss_list[i]), session_symptoms_28(ss_list[i + 1])
            delta_norms.append(np.linalg.norm(s2 - s1))

    if delta_norms:
        delta_norms = np.array(delta_norms)
        print(f"    연속 회기 간 28-증상 변화의 L2 norm 분포 (n={len(delta_norms)} 쌍):")
        print(f"      mean={delta_norms.mean():.4f}, median={np.median(delta_norms):.4f}")
        print(f"      [0% 변화 비율]: {(delta_norms < 0.01).mean():.4f}")
        print(f"      유의미한 변화(>0.1) 비율: {(delta_norms > 0.1).mean():.4f}")
    else:
        print("    종단 데이터 없음 (한 환자가 모두 1회기뿐)")

    # ----- 환자 ID 패턴 분석 -----
    print(f"\n[6] 환자 ID 패턴 (예시 20개)")
    examples = list(patient_to_diag.keys())[:20]
    for pid in examples:
        n = patient_session_count[pid]
        print(f"      {pid:8s} → {patient_to_diag[pid]:12s} ({n}회기 진행)")

    print(f"\n{'='*70}")
    print(f"  검증 완료: {len(sessions):,}개 세션, {len(patient_to_diag):,}명 환자")
    print(f"{'='*70}\n")

    return {
        'n_sessions': len(sessions),
        'n_patients': len(patient_to_diag),
        'diagnosis_session': dict(diag_session),
        'diagnosis_patient': dict(diag_patient),
    }


def main():
    parser = argparse.ArgumentParser(description="#58 데이터 무결성 검증")
    parser.add_argument('--extracted', '-e', required=True, type=str,
                        help='압축 해제 디렉토리 (예: ~/projects/k-ahfm-clinic/data/extracted)')
    parser.add_argument('--split', '-s', default='training', choices=['training', 'validation'],
                        help='검증할 split (기본: training)')
    args = parser.parse_args()

    extracted_dir = Path(args.extracted).expanduser().resolve()
    verify(extracted_dir, split=args.split)


if __name__ == '__main__':
    main()
