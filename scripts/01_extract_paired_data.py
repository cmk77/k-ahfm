#!/usr/bin/env python3
"""
#539 multimodal 데이터에서 paired (얼굴 crop, 텍스트, 감정 라벨) 추출.

각 clip_X.json + clip_X.mp4 처리:
  1. JSON에서 frame별 person 라벨 추출
  2. paired 조건 (bbox + image_emotion + text) 만족하는 person만 선택
  3. MP4에서 해당 frame 추출 → bbox로 얼굴 crop → 224x224 저장
  4. metadata JSON에 (face_path, text, image_emotion, text_emotion, ...) 기록

산출:
    {output_dir}/
    ├── faces/
    │   ├── clip0001_f0128_p1.jpg     (224x224 얼굴 crop)
    │   ├── clip0001_f0128_p2.jpg
    │   └── ...
    └── pairs.jsonl                    (paired metadata)

사용법:
    python scripts/01_extract_paired_data.py \\
        --root <외부 데이터 경로>/aihub_539_multimodal \\
        --output <repo>/data/phase2_pairs \\
        --max-clips 200 \\
        --max-pairs-per-clip 50
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def find_clip_dirs(root: Path):
    """539 multimodal root에서 모든 clip_X 디렉토리 발견."""
    # 구조: root/<range_folder>/<range_folder>/clip_X/clip_X.{json,mp4}
    # 또는: root/<range_folder>/clip_X/...
    clips = []
    for json_path in root.rglob('clip_*.json'):
        clip_dir = json_path.parent
        clip_id_str = json_path.stem.replace('clip_', '')
        try:
            clip_id = int(clip_id_str)
            mp4_path = clip_dir / f'clip_{clip_id}.mp4'
            if mp4_path.exists():
                clips.append({'clip_id': clip_id, 'json': json_path, 'mp4': mp4_path})
        except ValueError:
            continue
    clips.sort(key=lambda c: c['clip_id'])
    return clips


def extract_paired_persons(clip_json_data):
    """단일 clip JSON에서 paired (bbox + image_emotion + text) person 추출."""
    pairs = []
    for frame_id_str, frame_data in clip_json_data.get('data', {}).items():
        try:
            frame_num = int(frame_id_str)
        except ValueError:
            continue
        if not isinstance(frame_data, dict):
            continue

        for person_id, person_data in frame_data.items():
            if not isinstance(person_data, dict):
                continue
            if person_data.get('label') != 'person':
                continue

            # 필수: bounding box
            try:
                xtl = float(person_data['xtl'])
                ytl = float(person_data['ytl'])
                xbr = float(person_data['xbr'])
                ybr = float(person_data['ybr'])
            except (KeyError, ValueError):
                continue

            # 필수: image emotion
            img_emo = person_data.get('emotion', {}).get('image', {}).get('emotion')
            if not img_emo:
                continue

            # 필수: text utterance (paired condition)
            text_script = person_data.get('text', {}).get('script')
            if not text_script:
                continue

            # 추가 metadata (있으면 저장)
            text_emo = person_data.get('emotion', {}).get('text', {}).get('emotion')
            multimodal_emo = person_data.get('emotion', {}).get('multimodal', {}).get('emotion')
            valence_img = person_data.get('emotion', {}).get('image', {}).get('valence')
            arousal_img = person_data.get('emotion', {}).get('image', {}).get('arousal')

            pairs.append({
                'frame_num': frame_num,
                'person_id': person_data.get('person_id', str(person_id)),
                'bbox': [xtl, ytl, xbr, ybr],
                'text': text_script,
                'image_emotion': img_emo,
                'text_emotion': text_emo,
                'multimodal_emotion': multimodal_emo,
                'valence_image': valence_img,
                'arousal_image': arousal_img,
            })
    return pairs


def crop_face_from_frame(frame, bbox, padding_ratio=0.15, output_size=(224, 224)):
    """frame과 bbox로 얼굴 crop + 224x224 resize.

    padding_ratio: bbox 주변 패딩 비율 (배경 포함하여 face가 잘리지 않게)
    """
    H, W = frame.shape[:2]
    xtl, ytl, xbr, ybr = bbox
    bw, bh = xbr - xtl, ybr - ytl

    # 패딩 추가
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1 = max(0, int(xtl - pad_x))
    y1 = max(0, int(ytl - pad_y))
    x2 = min(W, int(xbr + pad_x))
    y2 = min(H, int(ybr + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # 정사각형으로 만들기 (긴 축 기준 padding)
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    pad_h = side - ch
    pad_w = side - cw
    crop_square = cv2.copyMakeBorder(
        crop, pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_REPLICATE,
    )

    resized = cv2.resize(crop_square, output_size, interpolation=cv2.INTER_AREA)
    return resized


def process_clip(clip_info, output_dir, max_pairs_per_clip=None, jpeg_quality=85):
    """단일 clip 처리: paired data 추출 + face crop 저장."""
    with open(clip_info['json'], 'r', encoding='utf-8') as f:
        clip_data = json.load(f)
    clip_id = int(clip_data.get('clip_id', clip_info['clip_id']))

    # JSON에서 paired person 추출
    paired = extract_paired_persons(clip_data)
    if not paired:
        return []

    # MP4 열기
    cap = cv2.VideoCapture(str(clip_info['mp4']))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 같은 frame 다중 person 처리를 위해 frame_num별로 그룹화
    by_frame = defaultdict(list)
    for p in paired:
        by_frame[p['frame_num']].append(p)
    frame_nums_sorted = sorted(by_frame.keys())

    # max_pairs_per_clip 적용 (uniform sampling)
    if max_pairs_per_clip is not None and len(paired) > max_pairs_per_clip:
        # uniform sample of frames
        stride = max(1, len(frame_nums_sorted) // max_pairs_per_clip)
        frame_nums_sorted = frame_nums_sorted[::stride][:max_pairs_per_clip]

    results = []
    faces_dir = output_dir / 'faces'
    faces_dir.mkdir(parents=True, exist_ok=True)

    for frame_num in frame_nums_sorted:
        if frame_num >= total_frames or frame_num < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        for person in by_frame[frame_num]:
            face = crop_face_from_frame(frame, person['bbox'])
            if face is None:
                continue
            face_filename = f"clip{clip_id:05d}_f{frame_num:06d}_p{person['person_id']}.jpg"
            face_path = faces_dir / face_filename
            cv2.imwrite(str(face_path), face,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

            result = dict(person)
            result['clip_id'] = clip_id
            result['face_path'] = str(face_path.relative_to(output_dir))
            results.append(result)

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="#539 paired text-face 추출")
    parser.add_argument('--root', required=True, type=str,
                       help='#539 multimodal root dir')
    parser.add_argument('--output', required=True, type=str,
                       help='Output dir for faces + pairs.jsonl')
    parser.add_argument('--max-clips', type=int, default=200,
                       help='Process at most N clips (default 200 for pilot)')
    parser.add_argument('--max-pairs-per-clip', type=int, default=50,
                       help='Sample at most N pairs per clip uniformly')
    parser.add_argument('--start-clip', type=int, default=1,
                       help='Start from clip ID')
    parser.add_argument('--jpeg-quality', type=int, default=85)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"#539 root: {root}")
    print(f"Output: {output_dir}")
    print(f"Max clips: {args.max_clips}, Max pairs/clip: {args.max_pairs_per_clip}")

    print(f"\n[1/2] Clip 디렉토리 검색 중...")
    all_clips = find_clip_dirs(root)
    print(f"  → {len(all_clips):,} clips 발견")

    if not all_clips:
        sys.exit(f"ERROR: clip_*.json/mp4 쌍이 발견되지 않음. --root 경로 확인.")

    # 처리할 clips
    target_clips = [c for c in all_clips if c['clip_id'] >= args.start_clip][:args.max_clips]
    print(f"  → 처리 대상: {len(target_clips):,} clips (id {args.start_clip}..{target_clips[-1]['clip_id']})")

    # 처리
    print(f"\n[2/2] 처리 + face crop 추출 중...")
    all_pairs = []
    img_emo_counter = Counter()
    text_emo_counter = Counter()
    n_clips_processed = 0
    n_clips_with_pairs = 0

    pairs_jsonl_path = output_dir / 'pairs.jsonl'
    with open(pairs_jsonl_path, 'w', encoding='utf-8') as fout:
        for clip_info in tqdm(target_clips, desc='clips', unit='clip'):
            try:
                pairs = process_clip(clip_info, output_dir,
                                     max_pairs_per_clip=args.max_pairs_per_clip,
                                     jpeg_quality=args.jpeg_quality)
                n_clips_processed += 1
                if pairs:
                    n_clips_with_pairs += 1
                for p in pairs:
                    fout.write(json.dumps(p, ensure_ascii=False) + '\n')
                    all_pairs.append(p)
                    img_emo_counter[p['image_emotion']] += 1
                    if p.get('text_emotion'):
                        text_emo_counter[p['text_emotion']] += 1
            except Exception as e:
                print(f"\n  [warn] clip_{clip_info['clip_id']} 처리 실패: {e}")
                continue

    # 통계
    print(f"\n{'='*70}")
    print(f"  ✓ 완료")
    print(f"{'='*70}")
    print(f"  Clips 처리:      {n_clips_processed:,} / 대상 {len(target_clips):,}")
    print(f"  Clips with paired: {n_clips_with_pairs:,}")
    print(f"  Total paired:    {len(all_pairs):,}")
    print(f"  Faces saved:     {len(all_pairs):,} (./faces/)")
    print(f"  Pairs metadata:  {pairs_jsonl_path}")
    print(f"\n  Image emotion 분포:")
    for e, c in img_emo_counter.most_common():
        pct = c / max(1, len(all_pairs)) * 100
        print(f"    {e:15s}: {c:6,} ({pct:.1f}%)")
    print(f"\n  Text emotion 분포 (있는 경우):")
    for e, c in text_emo_counter.most_common():
        print(f"    {e:15s}: {c:6,}")

    # Summary 저장
    summary = {
        'n_clips_processed': n_clips_processed,
        'n_clips_with_pairs': n_clips_with_pairs,
        'n_total_pairs': len(all_pairs),
        'image_emotion_dist': dict(img_emo_counter),
        'text_emotion_dist': dict(text_emo_counter),
        'args': vars(args),
    }
    with open(output_dir / 'extraction_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
