#!/usr/bin/env python3
"""
f_T 분류 헤드를 캐싱된 512-dim paragraph 임베딩에 후처리하여
세션 단위 emotion + V/A features 추출.

f_T 학습 시 사용된 두 헤드:
    - classifier_emo: Linear(512 → 8) for 8-class emotion
    - classifier_av:  Linear(512 → 2) for valence/arousal regression

f_T 모델 자체를 재실행하지 않고, 512-dim 임베딩에 헤드 가중치만 적용하여
빠르게 추출 (1,313 세션 약 1-2분).

산출:
    data/emotion_features.json
        {session_id: {
            emotion_softmax_mean: [8-dim, mean over paragraphs],
            emotion_softmax_max:  [8-dim, max-pool over paragraphs],
            valence_mean: float, valence_max: float,
            arousal_mean: float, arousal_max: float,
            entropy_mean: float (감정 분포 entropy 평균)
        }}

사용법:
    1) 헤드 키 자동 탐색:
       python scripts/extract_emotion_features.py \\
           --checkpoint checkpoints/f_T/best.pt \\
           --embeddings data/embeddings \\
           --output data/emotion_features.json \\
           --list-keys

    2) 자동 탐색된 키로 실행:
       python scripts/extract_emotion_features.py \\
           --checkpoint checkpoints/f_T/best.pt \\
           --embeddings data/embeddings \\
           --output data/emotion_features.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def discover_state_dict(checkpoint):
    """checkpoint dict 안에서 실제 state_dict 위치 자동 탐색."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ['model_state_dict', 'state_dict', 'model']:
        if key in checkpoint:
            sub = checkpoint[key]
            if isinstance(sub, dict) and any(hasattr(v, 'shape') for v in sub.values()):
                return sub
    return checkpoint


def find_head_keys(state_dict):
    """state_dict에서 emotion / av 헤드 키를 자동 탐색."""
    emo_w = emo_b = av_w = av_b = None

    # 1) 이름 기반 우선 검색 (emo / emotion / av)
    for k, v in state_dict.items():
        if not hasattr(v, 'shape'):
            continue
        kl = k.lower()
        if 'emo' in kl and 'weight' in kl and v.dim() == 2 and v.shape[0] == 8 and v.shape[1] in (512, 768):
            emo_w = k
        elif 'emo' in kl and 'bias' in kl and v.dim() == 1 and v.shape[0] == 8:
            emo_b = k
        elif 'av' in kl and 'weight' in kl and v.dim() == 2 and v.shape[0] == 2 and v.shape[1] in (512, 768):
            av_w = k
        elif 'av' in kl and 'bias' in kl and v.dim() == 1 and v.shape[0] == 2:
            av_b = k
        elif ('valence' in kl or 'arousal' in kl) and 'weight' in kl:
            if v.shape[0] == 2:
                av_w = k

    # 2) shape 기반 fallback
    if emo_w is None:
        for k, v in state_dict.items():
            if hasattr(v, 'shape') and v.dim() == 2 and v.shape[0] == 8 and v.shape[1] in (512, 768):
                emo_w = k
                emo_b_candidate = k.replace('weight', 'bias')
                if emo_b_candidate in state_dict and state_dict[emo_b_candidate].shape == (8,):
                    emo_b = emo_b_candidate
                break

    if av_w is None:
        for k, v in state_dict.items():
            if hasattr(v, 'shape') and v.dim() == 2 and v.shape[0] == 2 and v.shape[1] in (512, 768):
                av_w = k
                av_b_candidate = k.replace('weight', 'bias')
                if av_b_candidate in state_dict and state_dict[av_b_candidate].shape == (2,):
                    av_b = av_b_candidate
                break

    return {'emo_w': emo_w, 'emo_b': emo_b, 'av_w': av_w, 'av_b': av_b}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--list-keys', action='store_true',
                       help='state_dict 키 목록만 출력하고 종료')
    parser.add_argument('--emo-weight-key', type=str, default=None)
    parser.add_argument('--emo-bias-key', type=str, default=None)
    parser.add_argument('--av-weight-key', type=str, default=None)
    parser.add_argument('--av-bias-key', type=str, default=None)
    args = parser.parse_args()

    print(f"체크포인트 로드: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = discover_state_dict(checkpoint)
    print(f"state_dict 키 수: {len(sd)}")

    if args.list_keys:
        print("\n=== state_dict 키 목록 ===")
        for k, v in sd.items():
            shape = tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__
            print(f"  {k}: {shape}")
        print("\n자동 탐색 결과:")
        found = find_head_keys(sd)
        for name, key in found.items():
            print(f"  {name}: {key}")
        return

    # 헤드 키 결정
    if args.emo_weight_key and args.av_weight_key:
        keys = {
            'emo_w': args.emo_weight_key,
            'emo_b': args.emo_bias_key,
            'av_w': args.av_weight_key,
            'av_b': args.av_bias_key,
        }
    else:
        keys = find_head_keys(sd)

    print(f"\n사용할 헤드 키:")
    for name, key in keys.items():
        if key is not None:
            print(f"  {name} = {key}, shape={tuple(sd[key].shape)}")
        else:
            print(f"  {name} = (없음, 0으로 대체)")

    if keys['emo_w'] is None and keys['av_w'] is None:
        sys.exit("ERROR: emotion/av 헤드 모두 미발견. --list-keys로 확인하세요.")

    # 가중치 추출
    def get(key, default_shape):
        if key is None or key not in sd:
            return np.zeros(default_shape, dtype=np.float32)
        return sd[key].float().cpu().numpy()

    W_emo = get(keys['emo_w'], (8, 512))    # (8, 512)
    b_emo = get(keys['emo_b'], (8,))         # (8,)
    W_av = get(keys['av_w'], (2, 512))      # (2, 512)
    b_av = get(keys['av_b'], (2,))           # (2,)

    # 헤드의 입력 차원 확인 (512 또는 768 가능)
    in_dim_emo = W_emo.shape[1] if W_emo.shape[0] == 8 else 512
    in_dim_av = W_av.shape[1] if W_av.shape[0] == 2 else 512
    print(f"  → emotion head input_dim={in_dim_emo}, av head input_dim={in_dim_av}")

    # 캐싱된 임베딩 처리
    emb_dir = Path(args.embeddings)
    files = sorted(emb_dir.glob('*_para_emb.npy'))
    print(f"\n임베딩 파일 {len(files)}개 처리 시작...")

    results = {}
    all_v_raw, all_a_raw = [], []

    for i, f in enumerate(files):
        emb = np.load(f).astype(np.float32)  # (N_paragraphs, 512)
        if emb.shape[0] == 0:
            continue

        # 차원 호환성 검증
        if emb.shape[1] != in_dim_emo or emb.shape[1] != in_dim_av:
            if i == 0:
                print(f"  [경고] 임베딩 차원 {emb.shape[1]} vs head input {in_dim_emo}/{in_dim_av}")
                print(f"         그래도 진행. 차원 맞으면 결과 정상.")

        # Emotion logits
        if W_emo.shape[1] == emb.shape[1]:
            emo_logits = emb @ W_emo.T + b_emo  # (N, 8)
            # softmax
            emo_logits_shifted = emo_logits - emo_logits.max(axis=1, keepdims=True)
            emo_exp = np.exp(emo_logits_shifted)
            emo_softmax = emo_exp / (emo_exp.sum(axis=1, keepdims=True) + 1e-8)  # (N, 8)
        else:
            emo_softmax = np.ones((emb.shape[0], 8), dtype=np.float32) / 8

        # V/A predictions
        if W_av.shape[1] == emb.shape[1]:
            va_pred = emb @ W_av.T + b_av  # (N, 2)
            valence = va_pred[:, 0]
            arousal = va_pred[:, 1]
        else:
            valence = np.zeros(emb.shape[0], dtype=np.float32)
            arousal = np.zeros(emb.shape[0], dtype=np.float32)

        all_v_raw.append(valence)
        all_a_raw.append(arousal)

        # 세션 단위 집계 (mean + max-pool)
        session_id = f.stem.replace('_para_emb', '')
        # entropy of 8-class distribution per paragraph
        entropy_per_p = -(emo_softmax * np.log(emo_softmax + 1e-8)).sum(axis=1)

        results[session_id] = {
            'n_paragraphs': int(emb.shape[0]),
            'emotion_softmax_mean': emo_softmax.mean(axis=0).tolist(),  # (8,)
            'emotion_softmax_max': emo_softmax.max(axis=0).tolist(),     # (8,)
            'valence_raw_mean': float(valence.mean()),
            'valence_raw_max': float(valence.max()),
            'valence_raw_min': float(valence.min()),
            'arousal_raw_mean': float(arousal.mean()),
            'arousal_raw_max': float(arousal.max()),
            'arousal_raw_min': float(arousal.min()),
            'entropy_mean': float(entropy_per_p.mean()),
            'entropy_max': float(entropy_per_p.max()),
        }

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(files)}] processed")

    # 글로벌 V/A 정규화 (전체 데이터셋 기준 [0, 1])
    all_v = np.concatenate(all_v_raw) if all_v_raw else np.array([0.0])
    all_a = np.concatenate(all_a_raw) if all_a_raw else np.array([0.0])
    v_min, v_max = float(all_v.min()), float(all_v.max())
    a_min, a_max = float(all_a.min()), float(all_a.max())

    print(f"\nV/A 원본 범위:")
    print(f"  Valence: [{v_min:.3f}, {v_max:.3f}]")
    print(f"  Arousal: [{a_min:.3f}, {a_max:.3f}]")

    # 각 세션의 V/A를 [0, 1]로 정규화
    for sid, r in results.items():
        v_mean = (r['valence_raw_mean'] - v_min) / (v_max - v_min + 1e-8)
        v_max_s = (r['valence_raw_max'] - v_min) / (v_max - v_min + 1e-8)
        v_min_s = (r['valence_raw_min'] - v_min) / (v_max - v_min + 1e-8)
        a_mean = (r['arousal_raw_mean'] - a_min) / (a_max - a_min + 1e-8)
        a_max_s = (r['arousal_raw_max'] - a_min) / (a_max - a_min + 1e-8)
        a_min_s = (r['arousal_raw_min'] - a_min) / (a_max - a_min + 1e-8)
        r['valence_norm_mean'] = float(np.clip(v_mean, 0, 1))
        r['valence_norm_max'] = float(np.clip(v_max_s, 0, 1))
        r['valence_norm_min'] = float(np.clip(v_min_s, 0, 1))
        r['arousal_norm_mean'] = float(np.clip(a_mean, 0, 1))
        r['arousal_norm_max'] = float(np.clip(a_max_s, 0, 1))
        r['arousal_norm_min'] = float(np.clip(a_min_s, 0, 1))

    # 글로벌 통계 보고
    print(f"\n정규화 후 세션 V/A 분포:")
    v_means = [r['valence_norm_mean'] for r in results.values()]
    a_means = [r['arousal_norm_mean'] for r in results.values()]
    print(f"  Valence_mean: mean={np.mean(v_means):.3f}, std={np.std(v_means):.3f}")
    print(f"  Arousal_mean: mean={np.mean(a_means):.3f}, std={np.std(a_means):.3f}")

    # 저장 (메타데이터 포함)
    output = {
        '_meta': {
            'n_sessions': len(results),
            'v_min_raw': v_min, 'v_max_raw': v_max,
            'a_min_raw': a_min, 'a_max_raw': a_max,
            'emo_classes': 8,
            'emo_keys_used': keys,
        },
        'sessions': results,
    }

    print(f"\n저장: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    print(f"\n✓ 완료. 세션 {len(results)}개 emotion features 추출.")


if __name__ == '__main__':
    main()
