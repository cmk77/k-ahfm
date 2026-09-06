#!/usr/bin/env python
"""
06: 텍스트 감정 인코더 f_T 사전학습.

KLUE/RoBERTa-base를 #539 텍스트 corpus로 fine-tune. 학습 후 backbone과 
projection head를 저장하여 K-AHFM-Clinic의 Stage 1 (#58 paragraph 임베딩)에
사용한다.

학습 task:
    - 주: emotion.multimodal 8-class (focal loss with class weighting)
    - 보조 1: emotion.text 8-class
    - 보조 2: arousal/valence 회귀 MSE

실행 예:
    python 06_pretrain_text_emotion.py \\
        --corpus data/processed/aihub539_corpus.jsonl \\
        --output checkpoints/f_T \\
        --batch-size 64 --epochs 5
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.aihub539 import EMOTION_CLASSES, EMOTION_TO_INDEX, load_corpus_jsonl
from data.text_emotion_dataset import TextEmotionDataset, emotion_collate_fn
from models.text_encoder import (
    KoreanTextEmotionEncoder,
    TextEncoderConfig,
    multitask_loss,
)


# ============================================================================
# Helper
# ============================================================================
def compute_class_weights(labels: list, n_classes: int) -> torch.Tensor:
    """클래스 균형 가중치 계산 (inverse frequency)."""
    counter = Counter(labels)
    weights = torch.zeros(n_classes)
    total = sum(counter.values())
    for c in range(n_classes):
        n = counter.get(c, 1)  # 0 방지
        weights[c] = total / (n_classes * n)
    return weights


@torch.no_grad()
def evaluate(model, dataloader, device, n_classes: int) -> dict:
    """검증 데이터에서 평가 메트릭 계산."""
    model.eval()
    all_preds_mm = []
    all_labels_mm = []
    all_valence_pred, all_valence_true = [], []
    all_arousal_pred, all_arousal_true = [], []
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        emo_mm = batch['emotion_mm_label'].to(device)
        emo_text = batch['emotion_text_label'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)

        outputs = model(input_ids, attn)
        loss, _ = multitask_loss(outputs, {
            'emotion_mm_label': emo_mm,
            'emotion_text_label': emo_text,
            'valence': valence,
            'arousal': arousal,
        })
        total_loss += loss.item()
        n_batches += 1

        preds_mm = outputs['emotion_mm_logits'].argmax(dim=-1)
        valid = emo_mm != -100
        all_preds_mm.extend(preds_mm[valid].cpu().tolist())
        all_labels_mm.extend(emo_mm[valid].cpu().tolist())
        all_valence_pred.extend(outputs['valence_pred'].cpu().tolist())
        all_valence_true.extend(valence.cpu().tolist())
        all_arousal_pred.extend(outputs['arousal_pred'].cpu().tolist())
        all_arousal_true.extend(arousal.cpu().tolist())

    if not all_preds_mm:
        return {'loss': total_loss / max(n_batches, 1), 'accuracy': 0.0}

    # Accuracy
    preds = np.array(all_preds_mm)
    labels = np.array(all_labels_mm)
    accuracy = float((preds == labels).mean())

    # Per-class F1 (macro)
    from collections import defaultdict
    cls_tp = defaultdict(int)
    cls_fp = defaultdict(int)
    cls_fn = defaultdict(int)
    for p, t in zip(preds, labels):
        if p == t:
            cls_tp[t] += 1
        else:
            cls_fp[p] += 1
            cls_fn[t] += 1
    f1s = []
    for c in range(n_classes):
        prec = cls_tp[c] / max(cls_tp[c] + cls_fp[c], 1)
        rec = cls_tp[c] / max(cls_tp[c] + cls_fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    # AV Pearson 상관
    v_pred = np.array(all_valence_pred); v_true = np.array(all_valence_true)
    a_pred = np.array(all_arousal_pred); a_true = np.array(all_arousal_true)

    def _pearson(x, y):
        if len(x) < 2 or x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'valence_pearson': _pearson(v_pred, v_true),
        'arousal_pearson': _pearson(a_pred, a_true),
        'n_eval': len(all_preds_mm),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="텍스트 감정 인코더 f_T 사전학습")
    parser.add_argument('--corpus', required=True, type=str, help='발화 corpus JSONL')
    parser.add_argument('--output', required=True, type=str, help='체크포인트 출력 디렉토리')
    parser.add_argument('--backbone', default='klue/roberta-base', type=str)
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true', help='Mixed precision 사용')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-utterances', type=int, default=None,
                        help='디버깅용 최대 발화 수 (기본 None = 전체)')
    args = parser.parse_args()

    # 시드
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  06: 텍스트 감정 인코더 f_T 사전학습")
    print(f"{'='*70}")
    print(f"  Backbone:        {args.backbone}")
    print(f"  Device:          {device}")
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU:             {p.name} ({p.total_memory/1e9:.1f} GB)")
    print(f"  Embedding dim:   {args.embedding_dim}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  LR:              {args.lr}")
    print(f"  Mixed precision: {args.fp16}\n")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1. Corpus 로드 =====
    print(f"[1/6] Corpus 로드: {args.corpus}")
    corpus_path = Path(args.corpus).expanduser().resolve()
    utterances = load_corpus_jsonl(corpus_path)
    if args.max_utterances:
        utterances = utterances[:args.max_utterances]
    print(f"  → {len(utterances):,}개 발화")

    # ===== 2. Tokenizer & Model =====
    print(f"\n[2/6] Tokenizer & Model 로드: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    cfg = TextEncoderConfig(
        backbone_name=args.backbone,
        num_emotion_classes=len(EMOTION_CLASSES),
        embedding_dim=args.embedding_dim,
    )
    model = KoreanTextEmotionEncoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  → 학습 가능 파라미터: {n_params/1e6:.1f}M")

    # ===== 3. Dataset & Split =====
    print(f"\n[3/6] Dataset 구성")
    full_ds = TextEmotionDataset(utterances, tokenizer, max_length=args.max_length, require_primary=True)
    n_total = len(full_ds)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    print(f"  → train: {n_train:,}, val: {n_val:,}")

    # Class weights (inverse frequency)
    train_labels = [full_ds.utterances[i].emotion_multimodal for i in train_ds.indices]
    train_label_indices = [EMOTION_TO_INDEX[e] for e in train_labels if e in EMOTION_TO_INDEX]
    class_weights = compute_class_weights(train_label_indices, len(EMOTION_CLASSES)).to(device)
    print(f"  → class weights: {class_weights.cpu().tolist()}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=emotion_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=emotion_collate_fn,
        pin_memory=True,
    )

    # ===== 4. Optimizer & Scheduler =====
    print(f"\n[4/6] Optimizer & Scheduler")
    no_decay = ['bias', 'LayerNorm.weight']
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # ===== 5. Training Loop =====
    print(f"\n[5/6] 학습 시작")
    print(f"  total steps: {total_steps:,}, warmup: {warmup_steps:,}")

    best_val_acc = 0.0
    log_lines = []
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn = batch['attention_mask'].to(device, non_blocking=True)
            emo_mm = batch['emotion_mm_label'].to(device, non_blocking=True)
            emo_text = batch['emotion_text_label'].to(device, non_blocking=True)
            valence = batch['valence'].to(device, non_blocking=True)
            arousal = batch['arousal'].to(device, non_blocking=True)

            optimizer.zero_grad()

            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attn)
                    loss, _ = multitask_loss(
                        outputs,
                        {'emotion_mm_label': emo_mm,
                         'emotion_text_label': emo_text,
                         'valence': valence, 'arousal': arousal},
                        class_weights=class_weights,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attn)
                loss, _ = multitask_loss(
                    outputs,
                    {'emotion_mm_label': emo_mm,
                     'emotion_text_label': emo_text,
                     'valence': valence, 'arousal': arousal},
                    class_weights=class_weights,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_steps += 1
            if (step + 1) % 50 == 0:
                avg = running_loss / running_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  ep {epoch+1}/{args.epochs}  step {step+1}/{len(train_loader)}  "
                      f"loss {avg:.4f}  lr {lr:.2e}")

        epoch_time = time.time() - epoch_start

        # Validation
        val_metrics = evaluate(model, val_loader, device, len(EMOTION_CLASSES))
        log_msg = (f"\n  [epoch {epoch+1}] train loss {running_loss/running_steps:.4f}  "
                   f"val loss {val_metrics['loss']:.4f}  "
                   f"acc {val_metrics['accuracy']:.4f}  "
                   f"macro F1 {val_metrics['macro_f1']:.4f}  "
                   f"V-r {val_metrics['valence_pearson']:.3f}  "
                   f"A-r {val_metrics['arousal_pearson']:.3f}  "
                   f"({epoch_time:.0f}s)")
        print(log_msg)
        log_lines.append(log_msg.strip())

        # Best checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            ckpt_path = output_dir / 'best.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg.__dict__,
                'epoch': epoch + 1,
                'val_metrics': val_metrics,
            }, ckpt_path)
            print(f"  ✓ Best checkpoint 저장: {ckpt_path} (val_acc={best_val_acc:.4f})")

    # ===== 6. Save final =====
    print(f"\n[6/6] 최종 체크포인트 저장")
    final_path = output_dir / 'final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg.__dict__,
        'epoch': args.epochs,
    }, final_path)
    print(f"  → {final_path}")

    # 토크나이저도 저장 (재현성)
    tokenizer.save_pretrained(output_dir / 'tokenizer')
    print(f"  → tokenizer 저장: {output_dir/'tokenizer'}")

    # 학습 로그 저장
    log_path = output_dir / 'training_log.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"  → 학습 로그: {log_path}")

    # 최종 메트릭 JSON
    final_metrics = {
        'best_val_accuracy': best_val_acc,
        'final_epoch': args.epochs,
        'config': vars(args),
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  ✓ f_T 사전학습 완료 (best val acc: {best_val_acc:.4f})")
    print(f"{'='*70}\n")
    print(f"다음 단계: python 07_precompute_paragraph_embeddings.py "
          f"--checkpoint {output_dir}/best.pt\n")


if __name__ == '__main__':
    main()
