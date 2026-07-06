#!/usr/bin/env python3
"""
f_V (Vision Encoder) 학습 — Cross-modal contrastive + 감정 분류.

아키텍처:
    ResNet50 (ImageNet pretrained)
      ↓
    Adaptive Avg Pool → 2048-dim
      ↓
    Projection (2048 → 512, LayerNorm)  ← f_T와 같은 공동 임베딩 공간
      ↓
    ┌─ Emotion classifier (512 → N_classes)
    └─ Contrastive head (used directly as 512-dim)

Loss:
    L_total = α * L_contrastive + β * L_classification
    
    L_contrastive: InfoNCE
        - Positive: f_V(face_i)와 cached f_T(text_i) (같은 paired sample)
        - Negative: batch 내 다른 sample들
    
    L_classification: CrossEntropy on image emotion

사용법:
    python scripts/03_train_fv_contrastive.py \\
        --pairs <repo>/data/phase2_pairs/pairs.jsonl \\
        --ft-embeddings <repo>/data/phase2_pairs/ft_text_embeddings.npy \\
        --faces-dir <repo>/data/phase2_pairs \\
        --output <repo>/checkpoints/f_V \\
        --epochs 20 --batch-size 64 --lr 1e-4
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Dataset
# ============================================================================
class PairedDataset(Dataset):
    """Paired (face, cached f_T embedding, image emotion) Dataset."""

    def __init__(self, pairs, faces_root, ft_embeddings, emotion_to_idx, transform=None):
        self.pairs = pairs
        self.faces_root = Path(faces_root)
        self.ft_embeddings = ft_embeddings  # (N, 512) numpy
        self.emotion_to_idx = emotion_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        face_path = self.faces_root / p['face_path']
        img = Image.open(face_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        ft_emb = torch.from_numpy(self.ft_embeddings[idx]).float()
        emo_idx = self.emotion_to_idx[p['image_emotion']]

        return {
            'image': img,
            'ft_emb': ft_emb,
            'emotion': torch.tensor(emo_idx, dtype=torch.long),
            'clip_id': p['clip_id'],
        }


# ============================================================================
# f_V 모델
# ============================================================================
class fVModel(nn.Module):
    def __init__(self, num_emotion_classes: int, embed_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        # ResNet50 backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove final FC
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection to 512-dim (같은 공동 공간으로)
        self.projection = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        # Emotion classifier
        self.classifier_emo = nn.Linear(embed_dim, num_emotion_classes)

    def forward(self, x):
        feat = self.backbone(x)              # (B, 2048)
        emb = self.projection(feat)           # (B, 512)
        emo_logits = self.classifier_emo(emb)  # (B, N_classes)
        return {
            'embedding': emb,
            'emotion_logits': emo_logits,
        }


# ============================================================================
# Loss
# ============================================================================
def info_nce_loss(fv_emb, ft_emb, temperature: float = 0.07):
    """InfoNCE contrastive loss with text-image symmetric.

    fv_emb: (B, D)  f_V output
    ft_emb: (B, D)  cached f_T output (positive pair)

    같은 인덱스가 positive, 다른 모든 인덱스가 negative.
    """
    # L2 normalize
    fv_norm = F.normalize(fv_emb, dim=-1)
    ft_norm = F.normalize(ft_emb, dim=-1)

    # Similarity matrix: (B, B)
    sim = fv_norm @ ft_norm.t() / temperature

    # Cross-entropy: diagonal is positive
    B = fv_emb.size(0)
    targets = torch.arange(B, device=fv_emb.device)

    # 양방향 InfoNCE: image→text + text→image
    loss_i2t = F.cross_entropy(sim, targets)
    loss_t2i = F.cross_entropy(sim.t(), targets)
    loss = (loss_i2t + loss_t2i) / 2
    return loss


# ============================================================================
# 평가
# ============================================================================
@torch.no_grad()
def evaluate(model, loader, emotion_to_idx, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    fv_embs_all, ft_embs_all = [], []

    for batch in loader:
        images = batch['image'].to(device)
        ft_embs = batch['ft_emb'].to(device)
        emotions = batch['emotion'].to(device)

        out = model(images)
        preds = out['emotion_logits'].argmax(dim=-1)

        correct += (preds == emotions).sum().item()
        total += emotions.size(0)
        for p, e in zip(preds.cpu().numpy(), emotions.cpu().numpy()):
            per_class_total[int(e)] += 1
            if p == e:
                per_class_correct[int(e)] += 1

        fv_embs_all.append(out['embedding'].cpu())
        ft_embs_all.append(ft_embs.cpu())

    fv_embs = torch.cat(fv_embs_all)
    ft_embs = torch.cat(ft_embs_all)

    # Cross-modal retrieval accuracy
    fv_n = F.normalize(fv_embs, dim=-1)
    ft_n = F.normalize(ft_embs, dim=-1)
    sim = fv_n @ ft_n.t()  # (N, N)
    N = sim.size(0)

    # Top-1/5/10 retrieval (image→text and text→image)
    targets = torch.arange(N)
    ranks_i2t = []
    ranks_t2i = []
    for k in range(N):
        order = sim[k].argsort(descending=True)
        ranks_i2t.append((order == targets[k]).nonzero()[0].item())
        order = sim[:, k].argsort(descending=True)
        ranks_t2i.append((order == targets[k]).nonzero()[0].item())

    ranks_i2t = np.array(ranks_i2t)
    ranks_t2i = np.array(ranks_t2i)

    metrics = {
        'classification_acc': correct / max(total, 1),
        'per_class_acc': {emo: per_class_correct[idx] / max(per_class_total[idx], 1)
                          for emo, idx in emotion_to_idx.items() if per_class_total[idx] > 0},
        'retrieval_i2t_R1': float((ranks_i2t < 1).mean()),
        'retrieval_i2t_R5': float((ranks_i2t < 5).mean()),
        'retrieval_i2t_R10': float((ranks_i2t < 10).mean()),
        'retrieval_t2i_R1': float((ranks_t2i < 1).mean()),
        'retrieval_t2i_R5': float((ranks_t2i < 5).mean()),
        'retrieval_t2i_R10': float((ranks_t2i < 10).mean()),
        'retrieval_i2t_median_rank': float(np.median(ranks_i2t)),
        'retrieval_t2i_median_rank': float(np.median(ranks_t2i)),
    }
    return metrics


# ============================================================================
# 메인 학습
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True, type=str)
    parser.add_argument('--ft-embeddings', required=True, type=str)
    parser.add_argument('--faces-dir', required=True, type=str,
                       help='pairs.jsonl의 face_path가 상대 경로일 때 기준 디렉토리')
    parser.add_argument('--output', required=True, type=str,
                       help='checkpoint 출력 디렉토리')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lambda-contrastive', type=float, default=1.0)
    parser.add_argument('--lambda-classification', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}\n  f_V 학습 시작\n{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # 1) 데이터 로드
    print(f"\n[1/4] Paired 데이터 로드")
    pairs = []
    with open(args.pairs, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"  Pairs: {len(pairs):,}")

    ft_embeddings = np.load(args.ft_embeddings)
    print(f"  f_T embeddings: {ft_embeddings.shape}")
    assert len(pairs) == ft_embeddings.shape[0], "pairs와 f_T 임베딩 수 불일치"

    # 감정 클래스 사전
    emo_counter = Counter(p['image_emotion'] for p in pairs)
    print(f"\n  Image emotion 클래스 ({len(emo_counter)}개):")
    for e, c in emo_counter.most_common():
        print(f"    {e:15s}: {c:6,}")
    emotion_to_idx = {emo: idx for idx, emo in enumerate(sorted(emo_counter.keys()))}
    num_classes = len(emotion_to_idx)
    print(f"  num_classes: {num_classes}")

    # 2) Dataset / Loader
    print(f"\n[2/4] Dataset 구성")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Clip-based split (다른 clip의 sample만 val에 들어가도록 — leakage 방지)
    clip_ids = sorted(set(p['clip_id'] for p in pairs))
    rng = random.Random(args.seed)
    rng.shuffle(clip_ids)
    n_val_clips = max(1, int(len(clip_ids) * args.val_ratio))
    val_clip_ids = set(clip_ids[:n_val_clips])
    train_clip_ids = set(clip_ids[n_val_clips:])
    print(f"  Train clips: {len(train_clip_ids)}, Val clips: {len(val_clip_ids)}")

    train_indices = [i for i, p in enumerate(pairs) if p['clip_id'] in train_clip_ids]
    val_indices = [i for i, p in enumerate(pairs) if p['clip_id'] in val_clip_ids]

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    train_ft = ft_embeddings[train_indices]
    val_ft = ft_embeddings[val_indices]
    print(f"  Train pairs: {len(train_pairs):,}, Val pairs: {len(val_pairs):,}")

    train_ds = PairedDataset(train_pairs, args.faces_dir, train_ft, emotion_to_idx, train_transform)
    val_ds = PairedDataset(val_pairs, args.faces_dir, val_ft, emotion_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 3) Model
    print(f"\n[3/4] f_V 모델 초기화 (ResNet50)")
    model = fVModel(num_emotion_classes=num_classes, embed_dim=512, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}")
    print(f"  Trainable:    {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4) Training
    print(f"\n[4/4] 학습 시작")
    best_r1 = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_con = 0.0
        epoch_loss_cls = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch')
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            ft_embs = batch['ft_emb'].to(device, non_blocking=True)
            emotions = batch['emotion'].to(device, non_blocking=True)

            out = model(images)
            fv_emb = out['embedding']

            loss_con = info_nce_loss(fv_emb, ft_embs, args.temperature)
            loss_cls = F.cross_entropy(out['emotion_logits'], emotions)
            loss = args.lambda_contrastive * loss_con + args.lambda_classification * loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_total += loss.item()
            epoch_loss_con += loss_con.item()
            epoch_loss_cls += loss_cls.item()
            n_batches += 1
            pbar.set_postfix({
                'L': f'{loss.item():.3f}',
                'con': f'{loss_con.item():.3f}',
                'cls': f'{loss_cls.item():.3f}',
            })

        scheduler.step()

        avg_loss = epoch_loss_total / max(n_batches, 1)
        avg_con = epoch_loss_con / max(n_batches, 1)
        avg_cls = epoch_loss_cls / max(n_batches, 1)

        # Val
        val_metrics = evaluate(model, val_loader, emotion_to_idx, device)
        print(f"  Epoch {epoch+1:3d}: train_loss={avg_loss:.3f} (con={avg_con:.3f}, cls={avg_cls:.3f})  "
              f"val_acc={val_metrics['classification_acc']:.3f}  "
              f"R@1(i2t)={val_metrics['retrieval_i2t_R1']:.3f}  "
              f"R@5(i2t)={val_metrics['retrieval_i2t_R5']:.3f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_loss_con': avg_con,
            'train_loss_cls': avg_cls,
            'val_classification_acc': val_metrics['classification_acc'],
            'val_retrieval_i2t_R1': val_metrics['retrieval_i2t_R1'],
            'val_retrieval_i2t_R5': val_metrics['retrieval_i2t_R5'],
            'val_retrieval_i2t_R10': val_metrics['retrieval_i2t_R10'],
            'val_retrieval_t2i_R1': val_metrics['retrieval_t2i_R1'],
        })

        # 체크포인트 저장
        if val_metrics['retrieval_i2t_R1'] > best_r1:
            best_r1 = val_metrics['retrieval_i2t_R1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'emotion_to_idx': emotion_to_idx,
                'args': vars(args),
            }, output_dir / 'best.pt')
            print(f"    ✓ best.pt 저장 (R@1={best_r1:.3f})")

    # final.pt
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'emotion_to_idx': emotion_to_idx,
        'args': vars(args),
        'history': history,
    }, output_dir / 'final.pt')

    # history 저장
    with open(output_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\n  ✓ 학습 완료. Best val R@1: {best_r1:.3f}\n{'='*70}")
    print(f"  Checkpoint: {output_dir}/best.pt")


if __name__ == '__main__':
    main()
