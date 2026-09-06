#!/usr/bin/env python
"""
05: #539 멀티모달 영상 데이터에서 텍스트 발화 corpus 추출.

전체 clip JSON을 재귀 탐색하여 (1) 중복 제거된 unique 발화 추출, (2) 8-class
감정 라벨과 arousal/valence 회귀 라벨 정렬, (3) JSONL 형식으로 저장.

실행 예:
    python 05_build_539_corpus.py \\
        --source ~/datasets/aihub_539_multimodal \\
        --output data/processed/aihub539_corpus.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.aihub539 import (
    parse_all_clips,
    print_corpus_statistics,
    save_corpus_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description="#539 텍스트 발화 corpus 추출")
    parser.add_argument('--source', '-s', required=True, type=str,
                        help='#539 데이터 루트 (예: ~/datasets/aihub_539_multimodal)')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='출력 JSONL 경로 (예: data/processed/aihub539_corpus.jsonl)')
    parser.add_argument('--max-clips', type=int, default=None,
                        help='디버깅용 최대 clip 수 (기본 None = 전체)')
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  05: #539 텍스트 발화 corpus 추출")
    print(f"{'='*70}")
    print(f"  소스 디렉토리:   {source}")
    print(f"  출력 파일:       {output}\n")

    # 파싱
    utterances = parse_all_clips(source, progress=True, max_clips=args.max_clips)

    if not utterances:
        sys.exit("ERROR: 발화 추출 실패")

    # 통계 출력
    print_corpus_statistics(utterances)

    # JSONL 저장
    save_corpus_jsonl(utterances, output)

    print(f"\n{'='*70}")
    print(f"  ✓ Corpus 추출 완료: {len(utterances):,}개 발화")
    print(f"{'='*70}\n")
    print(f"다음 단계: python 06_pretrain_text_emotion.py --corpus {output}\n")


if __name__ == '__main__':
    main()
