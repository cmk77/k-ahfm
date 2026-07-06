#!/usr/bin/env python3
"""
Paired data의 텍스트들을 f_T로 인코딩하여 512-dim 임베딩 캐싱.

입력: pairs.jsonl (01에서 생성)
출력: ft_embeddings.npy + index.json
        - ft_embeddings.npy: (N, 512) float32
        - index.json: pair_idx → embedding row idx 매핑

f_V contrastive learning에서 텍스트 측 anchor로 사용.

사용법:
    python scripts/02_cache_ft_text_embeddings.py \\
        --pairs <repo>/data/phase2_pairs/pairs.jsonl \\
        --ft-checkpoint <repo>/checkpoints/f_T/best.pt \\
        --tokenizer-dir <repo>/checkpoints/f_T/tokenizer \\
        --output <repo>/data/phase2_pairs/ft_text_embeddings.npy \\
        --max-length 96 --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True, type=str)
    parser.add_argument('--ft-checkpoint', required=True, type=str)
    parser.add_argument('--tokenizer-dir', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--max-length', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1) Pairs 로드
    print(f"\n[1/4] Pairs 로드: {args.pairs}")
    texts = []
    with open(args.pairs, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                texts.append(obj['text'])
    print(f"  → {len(texts):,} 텍스트")

    if not texts:
        sys.exit("ERROR: pairs.jsonl에서 텍스트를 추출하지 못함")

    # 2) f_T 모델 로드
    print(f"\n[2/4] f_T 모델 로드")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    # f_T 체크포인트 로드
    checkpoint = torch.load(args.ft_checkpoint, map_location='cpu', weights_only=False)
    sd = checkpoint
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                sd = checkpoint[key]
                break

    # 키 구조 자동 탐색: backbone (KLUE/RoBERTa) + projection
    backbone_keys = [k for k in sd.keys() if 'backbone' in k.lower() or 'encoder.layer' in k.lower() or 'roberta' in k.lower()]
    proj_keys = [k for k in sd.keys() if ('proj' in k.lower() or 'embedding' in k.lower()) and 'weight' in k.lower()
                 and len(sd[k].shape) == 2]

    print(f"  Backbone 관련 키: {len(backbone_keys)}개")
    print(f"  Projection 후보:")
    for k in proj_keys[:5]:
        print(f"    {k}: {tuple(sd[k].shape)}")

    # KLUE/RoBERTa backbone 로드 (이름 기반)
    base_name = 'klue/roberta-base'
    print(f"  Backbone 로드: {base_name}")
    backbone = AutoModel.from_pretrained(base_name)

    # 새 state_dict 구성: backbone 부분만 추출
    backbone_sd = {}
    for k, v in sd.items():
        # 다양한 prefix 시도
        for prefix in ['backbone.', 'model.', 'encoder.']:
            if k.startswith(prefix):
                clean_key = k[len(prefix):]
                backbone_sd[clean_key] = v
                break
        else:
            # backbone일 가능성: roberta/bert prefix 또는 embeddings/encoder.layer 등
            if k.startswith(('roberta.', 'bert.')):
                clean_key = k.split('.', 1)[1]
                backbone_sd[clean_key] = v
            elif k.startswith(('embeddings.', 'encoder.layer.', 'pooler.')):
                backbone_sd[k] = v

    if backbone_sd:
        missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
        print(f"  backbone 로드: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"  [경고] checkpoint에서 backbone 가중치 추출 실패. 베이스 모델 그대로 사용.")

    backbone = backbone.to(device).eval()

    # Projection layer 찾기
    proj_w_key = proj_b_key = norm_w_key = norm_b_key = None
    for k in sd.keys():
        kl = k.lower()
        if 'proj' in kl and 'weight' in kl and len(sd[k].shape) == 2:
            shp = sd[k].shape
            if shp[0] == 512 and shp[1] in (768, 1024):
                proj_w_key = k
                proj_b_key = k.replace('weight', 'bias')
                # 같은 prefix의 layernorm 찾기
                prefix = k.rsplit('.', 1)[0].rsplit('.', 1)[0]
                for k2 in sd.keys():
                    if k2.startswith(prefix) and 'norm' in k2.lower() and 'weight' in k2:
                        norm_w_key = k2
                        norm_b_key = k2.replace('weight', 'bias')
                        break
                break

    if proj_w_key is None:
        sys.exit(
            "ERROR: f_T projection layer (768→512) 미발견.\n"
            "       체크포인트 키 목록을 출력하여 직접 명시 필요."
        )

    print(f"  Projection: {proj_w_key} {tuple(sd[proj_w_key].shape)}")
    W_proj = sd[proj_w_key].to(device).float()
    b_proj = sd[proj_b_key].to(device).float() if proj_b_key in sd else torch.zeros(512, device=device)

    if norm_w_key and norm_w_key in sd:
        ln_w = sd[norm_w_key].to(device).float()
        ln_b = sd[norm_b_key].to(device).float() if norm_b_key in sd else torch.zeros(512, device=device)
        print(f"  LayerNorm: {norm_w_key}")
    else:
        ln_w = ln_b = None
        print(f"  LayerNorm: (없음)")

    # 3) 텍스트 인코딩
    print(f"\n[3/4] 텍스트 임베딩 추출 (batch_size={args.batch_size})")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc='encoding'):
            batch_texts = texts[i:i + args.batch_size]
            enc = tokenizer(batch_texts, max_length=args.max_length,
                           padding=True, truncation=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = backbone(**enc)
            # [CLS] token (first token of last hidden state)
            cls = out.last_hidden_state[:, 0, :]  # (B, 768)
            # Projection
            emb = torch.nn.functional.linear(cls, W_proj, b_proj)  # (B, 512)
            # LayerNorm if available
            if ln_w is not None:
                emb = torch.nn.functional.layer_norm(emb, [512], weight=ln_w, bias=ln_b)
            all_embeddings.append(emb.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  → embeddings shape: {embeddings.shape}")

    # 4) 저장
    print(f"\n[4/4] 저장: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    # Index 메타
    index = {
        'n_pairs': len(texts),
        'embedding_dim': embeddings.shape[1],
        'max_length': args.max_length,
        'pairs_file': args.pairs,
        'ft_checkpoint': args.ft_checkpoint,
    }
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 완료. {len(texts):,} 텍스트 임베딩 저장.")
    print(f"  npy mean/std: {embeddings.mean():.4f} / {embeddings.std():.4f}")


if __name__ == '__main__':
    main()
