#!/usr/bin/env python
"""
AI Hub #58 심리상담 데이터 압축 해제 자동화 스크립트.

소스 디렉토리 구조 (자동 감지):
  ~/datasets/aihub_58/
    [Training | training_data | 1.Training] /
      [원천데이터 | original_data | source_data] /
        TS_001. 우울증_0001. 1회기.zip
        TS_001. 우울증_0002. 2회기.zip
        ...
      [라벨링데이터 | labeling_data | label_data] /
        TL_001. 우울증_0001. 1회기.zip
        ...
    [Validation | validation_data | 2.Validation] /
      (동일 구조)

출력 디렉토리 구조:
  <dest>/
    training/
      original/<diagnosis>/session_<NN>/
        resource_<diagnosis>_<N>_check_<patient_id>.txt
      labeling/<diagnosis>/session_<NN>/
        label_<diagnosis>_<N>_check_<patient_id>.json
    validation/ (동일)

특이사항:
- ZIP 내부 파일명 한글 인코딩 (CP949) 자동 처리
- 진단군 (우울증/불안장애/중독/일반군) 자동 파싱
- 회기 번호 자동 추출
- 실행 중간 중단 시 재실행 안전 (이미 해제된 파일은 건너뜀)
"""

import argparse
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# 프로젝트 src 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from data.constants import DIAGNOSIS_KOR_TO_ENG  # noqa: E402

# ============================================================================
# 정규식 / 패턴
# ============================================================================
ZIP_DIAGNOSIS_PATTERN = re.compile(r'(우울증|불안장애|중독|일반군|정상)')
ZIP_SESSION_PATTERN = re.compile(r'(\d+)회기')

# ============================================================================
# Korean filename decoder
# ============================================================================
def decode_korean_zipname(raw_name: str) -> str:
    """ZIP archive 내부 파일명을 CP949로 디코딩 (Windows에서 압축된 한글 zip 호환)."""
    if isinstance(raw_name, bytes):
        candidates = [raw_name]
    else:
        # zipfile은 CP437로 decode된 str을 줌 — 원래 bytes로 복원
        try:
            candidates = [raw_name.encode('cp437')]
        except UnicodeEncodeError:
            return raw_name
    for raw in candidates:
        for enc in ('cp949', 'euc-kr', 'utf-8'):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    return raw_name


# ============================================================================
# ZIP 파일명 파서
# ============================================================================
def parse_zip_filename(zip_name: str):
    """ZIP 파일명에서 (진단군 영문명, 회기번호) 추출.

    예: 'TL_001. 우울증_0001. 1회기.zip' → ('depression', 1)
    """
    diagnosis_kor = None
    m = ZIP_DIAGNOSIS_PATTERN.search(zip_name)
    if m:
        diagnosis_kor = m.group(1)
    if diagnosis_kor not in DIAGNOSIS_KOR_TO_ENG:
        return None, None
    diagnosis_eng = DIAGNOSIS_KOR_TO_ENG[diagnosis_kor]

    m = ZIP_SESSION_PATTERN.search(zip_name)
    session_num = int(m.group(1)) if m else None
    return diagnosis_eng, session_num


# ============================================================================
# 소스 디렉토리 자동 탐색
# ============================================================================
def detect_data_layout(source_dir: Path) -> dict:
    """소스 디렉토리에서 (split, kind) → 디렉토리 경로 매핑 자동 감지."""
    layout = {}

    # 단계 1: training/validation 구분 디렉토리 찾기
    split_aliases = {
        'training': ['Training', 'training', 'training_data', '1.Training', 'TR', '학습'],
        'validation': ['Validation', 'validation', 'validation_data', '2.Validation', 'VAL', 'VL', '검증', '평가'],
    }
    split_dirs = {}
    for entry in source_dir.iterdir():
        if not entry.is_dir():
            continue
        for split, aliases in split_aliases.items():
            if any(alias in entry.name for alias in aliases):
                split_dirs[split] = entry
                break

    # split이 발견되지 않은 경우, source_dir 자체에 original/labeling이 있는지 확인
    if not split_dirs:
        # source_dir 직속에 original/labeling이 있다면 training으로 간주
        for entry in source_dir.iterdir():
            if entry.is_dir() and ('원천' in entry.name or 'original' in entry.name.lower() or 'source' in entry.name.lower()):
                split_dirs['training'] = source_dir
                break

    # 단계 2: 각 split 하위에서 original/labeling 디렉토리 찾기
    kind_aliases = {
        'original': ['원천데이터', 'original_data', 'source_data', 'original', 'TS', '원천', 'Source'],
        'labeling': ['라벨링데이터', 'labeling_data', 'label_data', 'labeling', 'TL', '라벨', 'Label'],
    }
    for split, split_path in split_dirs.items():
        layout[split] = {}
        for entry in split_path.iterdir():
            if not entry.is_dir():
                continue
            for kind, aliases in kind_aliases.items():
                if any(alias in entry.name for alias in aliases):
                    layout[split][kind] = entry
                    break

    return layout


# ============================================================================
# ZIP 추출
# ============================================================================
def extract_zip(zip_path: Path, dest_dir: Path, skip_existing: bool = True) -> int:
    """ZIP 파일을 압축 해제하고, 추출된 파일 개수를 반환."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                korean_name = decode_korean_zipname(info.filename)
                base_name = os.path.basename(korean_name)
                if not base_name:
                    continue
                target = dest_dir / base_name
                if skip_existing and target.exists() and target.stat().st_size > 0:
                    count += 1
                    continue
                with zf.open(info) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                count += 1
    except zipfile.BadZipFile:
        print(f"  ERROR: 손상된 ZIP - {zip_path}", file=sys.stderr)
        return 0
    return count


# ============================================================================
# 메인
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AI Hub #58 심리상담 데이터 압축 해제",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--source', '-s', required=True, type=str,
                        help='AI Hub #58 데이터 루트 (예: ~/datasets/aihub_58)')
    parser.add_argument('--dest', '-d', required=True, type=str,
                        help='압축 해제 대상 (예: ~/projects/k-ahfm-clinic/data/extracted)')
    parser.add_argument('--dry-run', action='store_true',
                        help='실제 압축 해제 없이 계획만 출력')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='이미 해제된 파일도 덮어쓰기 (기본: 건너뜀)')
    args = parser.parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    dest_dir = Path(args.dest).expanduser().resolve()

    if not source_dir.exists():
        sys.exit(f"ERROR: 소스 디렉토리를 찾을 수 없음: {source_dir}")

    print(f"\n소스 디렉토리: {source_dir}")
    print(f"대상 디렉토리: {dest_dir}\n")

    # 디렉토리 구조 자동 감지
    print("== 1단계: 소스 디렉토리 구조 자동 감지 ==")
    layout = detect_data_layout(source_dir)
    if not layout:
        print("\n현재 소스 디렉토리 내용:")
        for entry in sorted(source_dir.iterdir()):
            print(f"  {entry.name}")
        sys.exit("\nERROR: Training/Validation × original/labeling 구조를 감지하지 못함.\n"
                 "디렉토리 이름을 다음 중 하나로 통일하거나 --source 경로를 확인하세요:\n"
                 "  Training | training_data | 1.Training\n"
                 "  원천데이터 | original_data | source_data\n"
                 "  라벨링데이터 | labeling_data | label_data")

    for split, parts in layout.items():
        print(f"  [{split}]")
        for kind, path in parts.items():
            n_zip = len(list(path.glob('*.zip')))
            print(f"    {kind:10s}: {path} ({n_zip} zip files)")

    # 압축 해제 계획 수립
    print("\n== 2단계: 압축 해제 계획 수립 ==")
    plan = []  # [(zip_path, dest_path, split, kind), ...]
    counts = defaultdict(lambda: defaultdict(int))
    parse_failed = []

    for split, parts in layout.items():
        for kind, src_path in parts.items():
            for zip_file in sorted(src_path.glob('*.zip')):
                diagnosis, session = parse_zip_filename(zip_file.name)
                if diagnosis is None:
                    parse_failed.append(zip_file.name)
                    continue
                session_str = f"session_{session:02d}" if session else "session_unknown"
                target_dir = dest_dir / split / kind / diagnosis / session_str
                plan.append((zip_file, target_dir, split, kind))
                counts[f"{split}/{kind}"][diagnosis] += 1

    print(f"  계획된 ZIP 해제: {len(plan)}개")
    if parse_failed:
        print(f"  WARN: 파일명 파싱 실패 {len(parse_failed)}개:")
        for fn in parse_failed[:5]:
            print(f"    {fn}")
        if len(parse_failed) > 5:
            print(f"    ... (외 {len(parse_failed) - 5}개)")

    print("\n  진단군별 ZIP 개수:")
    for split_kind in sorted(counts.keys()):
        diag_counts = counts[split_kind]
        total = sum(diag_counts.values())
        breakdown = ", ".join(f"{d}:{n}" for d, n in sorted(diag_counts.items()))
        print(f"    {split_kind:25s} {total:4d}  ({breakdown})")

    if args.dry_run:
        print("\n[--dry-run] 실제 해제 없이 종료")
        return

    # 압축 해제 실행
    print("\n== 3단계: 압축 해제 실행 ==")
    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit("ERROR: tqdm 패키지가 필요. setup_env.sh 먼저 실행")

    total_files = 0
    skip_existing = not args.no_skip_existing
    for zip_path, target_dir, split, kind in tqdm(plan, desc="ZIP 해제"):
        n = extract_zip(zip_path, target_dir, skip_existing=skip_existing)
        total_files += n

    print(f"\n  추출 완료: 총 {total_files:,}개 파일")
    print(f"  대상: {dest_dir}")

    # 무결성 빠른 체크
    print("\n== 4단계: 디렉토리 구조 검증 ==")
    for split in layout.keys():
        for kind in ['original', 'labeling']:
            split_kind_dir = dest_dir / split / kind
            if not split_kind_dir.exists():
                continue
            ext = '.txt' if kind == 'original' else '.json'
            file_count = sum(1 for _ in split_kind_dir.rglob(f'*{ext}'))
            print(f"  {split}/{kind:10s}: {file_count:,}개 {ext}")

    print("\n다음 단계:")
    print(f"  python 02_verify_data.py --extracted {dest_dir}\n")


if __name__ == '__main__':
    main()
