#!/usr/bin/env python
"""
07: #58 paragraph 임베딩 사전계산 및 디스크 캐싱.

학습된 f_T를 사용하여 모든 #58 paragraph의 임베딩을 한 번 계산하고
디스크에 캐싱한다. 이렇게 하면 K-AHFM-Clinic 본 학습과 모든 baseline·
ablation 실험에서 KLUE 백본 forward를 반복 계산하지 않아도 되어
학습 속도가 수십 배 빨라진다.

저장 형식:
    data/embeddings/<patient_id>_s<NN>_para_emb.npy
        shape: (num_paragraphs, embedding_dim)
        dtype: float32

실행 예:
    python 07_precompute_paragraph_embeddings.py \\
        --sessions data/processed/sessions.pkl \\
        --checkpoint checkpoints/f_T/best.pt \\
        --output data/embeddings \\
        --batch-size 64
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.parser import parse_all_sessions
from models.text_encoder import KoreanTextEmotionEncoder, TextEncoderConfig


# ============================================================================
# Helpers
# ============================================================================
class ParagraphBatchDataset(Dataset):
    """단일 세션의 paragraph들을 batch 단위로 처리하기 위한 임시 Dataset."""
    def __init__(self, paragraph_texts, tokenizer, max_length=128):
        self.texts = paragraph_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] if self.texts[idx] else ' '  # 빈 텍스트 방지
        enc = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


def collate_pad(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="#58 paragraph 임베딩 사전계산")
    parser.add_argument('--sessions', type=str, default=None,
                        help='기존 sessions.pkl (없으면 --extracted에서 새로 파싱)')
    parser.add_argument('--extracted', type=str, default=None,
                        help='--sessions 없을 시 압축 해제 디렉토리')
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'both'])
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='f_T 체크포인트 (예: checkpoints/f_T/best.pt)')
    parser.add_argument('--output', required=True, type=str,
                        help='임베딩 캐시 디렉토리 (예: data/embeddings)')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max-paragraphs-per-session', type=int, default=500,
                        help='세션당 처리할 최대 paragraph 수 (기본 500)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='이미 저장된 임베딩은 건너뜀')
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  07: #58 paragraph 임베딩 사전계산")
    print(f"{'='*70}")
    print(f"  체크포인트:      {args.checkpoint}")
    print(f"  출력 디렉토리:   {output_dir}")
    print(f"  Device:          {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU:             {p.name}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Mixed precision: {args.fp16}\n")

    # ===== 1. Sessions 로드 =====
    print(f"[1/4] Sessions 로드")
    if args.sessions and Path(args.sessions).exists():
        with open(args.sessions, 'rb') as f:
            sessions = pickle.load(f)
        print(f"  → {len(sessions):,}개 세션 로드 ({args.sessions})")
    elif args.extracted:
        extracted = Path(args.extracted).expanduser().resolve()
        splits_to_load = ['training', 'validation'] if args.split == 'both' else [args.split]
        sessions = []
        for sp in splits_to_load:
            labeling = extracted / sp / 'labeling'
            if labeling.exists():
                sessions.extend(parse_all_sessions(labeling, progress=True))
        print(f"  → {len(sessions):,}개 세션 파싱")
    else:
        sys.exit("ERROR: --sessions 또는 --extracted 둘 중 하나 필수")

    if not sessions:
        sys.exit("ERROR: 사용 가능한 세션 없음")

    # ===== 2. 체크포인트 로드 =====
    print(f"\n[2/4] f_T 체크포인트 로드")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = ckpt['config']
    cfg = TextEncoderConfig(**cfg_dict)
    model = KoreanTextEmotionEncoder(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  → 백본: {cfg.backbone_name}, embedding_dim={cfg.embedding_dim}")
    if 'val_metrics' in ckpt:
        print(f"  → val_acc: {ckpt['val_metrics'].get('accuracy', 'N/A')}")

    # Tokenizer
    tokenizer_dir = Path(args.checkpoint).parent / 'tokenizer'
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
    print(f"  → tokenizer 로드")

    # ===== 3. 세션별 임베딩 계산 + 저장 =====
    print(f"\n[3/4] 세션별 paragraph 임베딩 계산")
    from tqdm import tqdm

    total_paragraphs = 0
    n_skipped = 0
    n_processed = 0
    start = time.time()

    for s in tqdm(sessions, desc='세션'):
        if not s.patient_id or s.session_number <= 0:
            continue

        emb_path = output_dir / f"{s.patient_id}_s{s.session_number:02d}_para_emb.npy"
        if args.skip_existing and emb_path.exists():
            n_skipped += 1
            continue

        paragraphs = s.paragraphs[:args.max_paragraphs_per_session]
        if not paragraphs:
            # 빈 세션도 빈 배열 저장 (downstream에서 일관성)
            np.save(emb_path, np.zeros((0, cfg.embedding_dim), dtype=np.float32))
            n_processed += 1
            continue

        texts = [p.text if p.text else ' ' for p in paragraphs]

        # 작은 세션은 그냥 forward, 큰 세션은 mini-batch
        ds = ParagraphBatchDataset(texts, tokenizer, max_length=args.max_length)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_pad,
            pin_memory=True,
        )

        embs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attn = batch['attention_mask'].to(device, non_blocking=True)
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        emb = model.extract_embedding(input_ids, attn)
                else:
                    emb = model.extract_embedding(input_ids, attn)
                embs.append(emb.float().cpu().numpy())

        emb_array = np.concatenate(embs, axis=0).astype(np.float32)
        np.save(emb_path, emb_array)
        total_paragraphs += emb_array.shape[0]
        n_processed += 1

    elapsed = time.time() - start

    # ===== 4. 요약 =====
    print(f"\n[4/4] 완료 요약")
    print(f"  처리한 세션:     {n_processed:,}개")
    print(f"  건너뛴 세션:     {n_skipped:,}개")
    print(f"  총 임베딩 수:    {total_paragraphs:,}개 paragraph")
    print(f"  소요 시간:       {elapsed/60:.1f}분")
    print(f"  속도:            {total_paragraphs/elapsed:.0f} paragraph/s")

    # 디스크 사용량
    total_size = sum(f.stat().st_size for f in output_dir.glob('*_para_emb.npy'))
    print(f"  디스크 사용량:   {total_size/1e9:.2f} GB")

    print(f"\n{'='*70}")
    print(f"  ✓ 임베딩 사전계산 완료")
    print(f"{'='*70}\n")
    print(f"다음 단계: C-block (B1-B5 baseline 학습)\n")


if __name__ == '__main__':
    main()
