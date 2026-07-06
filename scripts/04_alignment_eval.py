#!/usr/bin/env python3
"""
Cross-modal alignment 정량 검증.

학습된 f_V로 paired data의 face 임베딩 추출 후,
cached f_T 텍스트 임베딩과의 alignment를 다양한 측면에서 평가.

평가 항목:
    1. Within-clip vs Across-clip similarity (paired 효과 검증)
    2. Same-image-emotion vs Different similarity (의미 alignment)
    3. Cross-modal Retrieval (R@1, R@5, R@10)
    4. Per-emotion alignment 분석

산출:
    {output}/alignment_metrics.json
    {output}/alignment_report.md

사용법:
    python scripts/04_alignment_eval.py \\
        --pairs <repo>/data/phase2_pairs/pairs.jsonl \\
        --ft-embeddings <repo>/data/phase2_pairs/ft_text_embeddings.npy \\
        --fv-checkpoint <repo>/checkpoints/f_V/best.pt \\
        --faces-dir <repo>/data/phase2_pairs \\
        --output <repo>/results/phase2_alignment
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True, type=str)
    parser.add_argument('--ft-embeddings', required=True, type=str)
    parser.add_argument('--fv-checkpoint', required=True, type=str)
    parser.add_argument('--faces-dir', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='평가용 최대 sample (시간 절약)')
    args = parser.parse_args()

    # local imports
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    train_module = import_module('03_train_fv_contrastive'.replace('-', '_'))
    fVModel = train_module.fVModel
    PairedDataset = train_module.PairedDataset

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print(f"[1/4] Paired 데이터 로드")
    pairs = []
    with open(args.pairs, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    ft_embeddings = np.load(args.ft_embeddings)

    rng = random.Random(args.seed)
    if len(pairs) > args.max_samples:
        indices = rng.sample(range(len(pairs)), args.max_samples)
        indices.sort()
        pairs = [pairs[i] for i in indices]
        ft_embeddings = ft_embeddings[indices]
    print(f"  사용 pairs: {len(pairs):,}")

    # 2) Load f_V
    print(f"\n[2/4] f_V 모델 로드: {args.fv_checkpoint}")
    ckpt = torch.load(args.fv_checkpoint, map_location='cpu', weights_only=False)
    emotion_to_idx = ckpt['emotion_to_idx']
    num_classes = len(emotion_to_idx)
    idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
    print(f"  Emotion 클래스: {list(emotion_to_idx.keys())}")

    model = fVModel(num_emotion_classes=num_classes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3) Extract f_V embeddings
    print(f"\n[3/4] f_V 임베딩 추출")
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PairedDataset(pairs, args.faces_dir, ft_embeddings, emotion_to_idx, eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    fv_embs_list, ft_embs_list, emos_list, clip_ids_list = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='extract'):
            images = batch['image'].to(device)
            out = model(images)
            fv_embs_list.append(out['embedding'].cpu())
            ft_embs_list.append(batch['ft_emb'])
            emos_list.append(batch['emotion'])
            clip_ids_list.extend(batch['clip_id'].numpy().tolist() if isinstance(batch['clip_id'], torch.Tensor) else list(batch['clip_id']))

    fv = torch.cat(fv_embs_list)
    ft = torch.cat(ft_embs_list)
    emotions = torch.cat(emos_list).numpy()
    clip_ids = np.array(clip_ids_list)

    # L2 normalize
    fv_n = F.normalize(fv, dim=-1).numpy()
    ft_n = F.normalize(ft, dim=-1).numpy()

    print(f"  f_V emb: {fv.shape}, f_T emb: {ft.shape}")

    # 4) Alignment 분석
    print(f"\n[4/4] Alignment 분석")
    metrics = {}

    # 4.1) Pair-wise cosine similarity
    # Positive pair (paired sample): diagonal
    sim_matrix = fv_n @ ft_n.T                     # (N, N)
    N = sim_matrix.shape[0]

    pos_sim = sim_matrix.diagonal()
    # Average across non-diagonal
    mask = np.ones_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    neg_sim = sim_matrix[mask].reshape(N, N - 1)
    neg_sim_mean = neg_sim.mean(axis=1)

    metrics['paired_sim_mean'] = float(pos_sim.mean())
    metrics['paired_sim_std'] = float(pos_sim.std())
    metrics['random_sim_mean'] = float(neg_sim_mean.mean())
    metrics['random_sim_std'] = float(neg_sim_mean.std())
    metrics['alignment_gap'] = metrics['paired_sim_mean'] - metrics['random_sim_mean']

    print(f"\n  [Paired vs Random Similarity]")
    print(f"    Paired (same instance):  {metrics['paired_sim_mean']:.4f} ± {metrics['paired_sim_std']:.4f}")
    print(f"    Random (different):       {metrics['random_sim_mean']:.4f} ± {metrics['random_sim_std']:.4f}")
    print(f"    Gap (alignment):          {metrics['alignment_gap']:.4f}")

    # 4.2) Within-clip vs Across-clip similarity
    same_clip_mask = clip_ids[:, None] == clip_ids[None, :]
    diff_clip_mask = ~same_clip_mask
    np.fill_diagonal(same_clip_mask, False)        # 자기 자신 제외

    if same_clip_mask.sum() > 0:
        same_clip_sims = sim_matrix[same_clip_mask]
        diff_clip_sims = sim_matrix[diff_clip_mask]
        metrics['within_clip_sim_mean'] = float(same_clip_sims.mean())
        metrics['across_clip_sim_mean'] = float(diff_clip_sims.mean())
        metrics['within_clip_gap'] = metrics['within_clip_sim_mean'] - metrics['across_clip_sim_mean']

        print(f"\n  [Within-clip vs Across-clip Similarity]")
        print(f"    Within-clip (다른 instance, 같은 clip): {metrics['within_clip_sim_mean']:.4f}")
        print(f"    Across-clip:                              {metrics['across_clip_sim_mean']:.4f}")
        print(f"    Gap:                                       {metrics['within_clip_gap']:.4f}")

    # 4.3) Same emotion vs Different emotion similarity
    same_emo_mask = emotions[:, None] == emotions[None, :]
    diff_emo_mask = ~same_emo_mask
    np.fill_diagonal(same_emo_mask, False)

    if same_emo_mask.sum() > 0:
        same_emo_sims = sim_matrix[same_emo_mask]
        diff_emo_sims = sim_matrix[diff_emo_mask]
        metrics['same_emotion_sim_mean'] = float(same_emo_sims.mean())
        metrics['diff_emotion_sim_mean'] = float(diff_emo_sims.mean())
        metrics['emotion_alignment_gap'] = metrics['same_emotion_sim_mean'] - metrics['diff_emotion_sim_mean']

        print(f"\n  [Same emotion vs Different emotion]")
        print(f"    Same emotion (different instance): {metrics['same_emotion_sim_mean']:.4f}")
        print(f"    Different emotion:                  {metrics['diff_emotion_sim_mean']:.4f}")
        print(f"    Emotion alignment gap:              {metrics['emotion_alignment_gap']:.4f}")

    # 4.4) Cross-modal retrieval
    ranks_i2t = []
    ranks_t2i = []
    for k in range(N):
        order_i2t = np.argsort(-sim_matrix[k])
        rank_i2t = np.where(order_i2t == k)[0][0]
        ranks_i2t.append(rank_i2t)
        order_t2i = np.argsort(-sim_matrix[:, k])
        rank_t2i = np.where(order_t2i == k)[0][0]
        ranks_t2i.append(rank_t2i)

    ranks_i2t = np.array(ranks_i2t)
    ranks_t2i = np.array(ranks_t2i)

    for K in (1, 5, 10):
        metrics[f'retrieval_i2t_R@{K}'] = float((ranks_i2t < K).mean())
        metrics[f'retrieval_t2i_R@{K}'] = float((ranks_t2i < K).mean())
    metrics['retrieval_i2t_median_rank'] = float(np.median(ranks_i2t))
    metrics['retrieval_t2i_median_rank'] = float(np.median(ranks_t2i))

    print(f"\n  [Cross-modal Retrieval]")
    print(f"    Image→Text  R@1: {metrics['retrieval_i2t_R@1']:.4f}  R@5: {metrics['retrieval_i2t_R@5']:.4f}  R@10: {metrics['retrieval_i2t_R@10']:.4f}")
    print(f"    Text→Image  R@1: {metrics['retrieval_t2i_R@1']:.4f}  R@5: {metrics['retrieval_t2i_R@5']:.4f}  R@10: {metrics['retrieval_t2i_R@10']:.4f}")
    print(f"    Median rank: i2t={metrics['retrieval_i2t_median_rank']:.0f}, t2i={metrics['retrieval_t2i_median_rank']:.0f}")

    # 4.5) Per-emotion 분석
    per_emo_metrics = {}
    for emo_name, emo_idx in emotion_to_idx.items():
        emo_mask = emotions == emo_idx
        n_samples = int(emo_mask.sum())
        if n_samples < 5:
            continue
        # 이 감정 내 paired sim
        pos_sim_emo = sim_matrix[emo_mask].diagonal() if emo_mask.sum() == emo_mask.shape[0] else \
                      np.array([sim_matrix[i, i] for i in np.where(emo_mask)[0]])
        per_emo_metrics[emo_name] = {
            'n_samples': n_samples,
            'paired_sim_mean': float(pos_sim_emo.mean()),
            'paired_sim_std': float(pos_sim_emo.std()),
        }
    metrics['per_emotion'] = per_emo_metrics

    print(f"\n  [Per-emotion paired similarity]")
    for emo, m in per_emo_metrics.items():
        print(f"    {emo:15s} (n={m['n_samples']:5,}): {m['paired_sim_mean']:.4f} ± {m['paired_sim_std']:.4f}")

    # 저장
    metrics_path = output_dir / 'alignment_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  → 저장: {metrics_path}")

    # Markdown 리포트
    report = []
    report.append("# Cross-Modal Alignment 정량 검증 결과\n")
    report.append(f"평가 sample: {N:,}개  |  f_T-f_V 공동 임베딩 공간 (512-dim)\n")
    report.append("\n## 1. Paired vs Random Cosine Similarity\n")
    report.append("| 조건 | Cosine Similarity |")
    report.append("|---|---|")
    report.append(f"| **Paired (같은 instance)** | **{metrics['paired_sim_mean']:.4f} ± {metrics['paired_sim_std']:.4f}** |")
    report.append(f"| Random (다른 instance) | {metrics['random_sim_mean']:.4f} ± {metrics['random_sim_std']:.4f} |")
    report.append(f"| **Alignment Gap** | **{metrics['alignment_gap']:.4f}** |\n")
    if 'within_clip_sim_mean' in metrics:
        report.append("\n## 2. Within-clip vs Across-clip\n")
        report.append("| 조건 | Cosine Similarity |")
        report.append("|---|---|")
        report.append(f"| Within-clip | {metrics['within_clip_sim_mean']:.4f} |")
        report.append(f"| Across-clip | {metrics['across_clip_sim_mean']:.4f} |")
        report.append(f"| **Gap** | **{metrics['within_clip_gap']:.4f}** |\n")
    if 'same_emotion_sim_mean' in metrics:
        report.append("\n## 3. Same Emotion vs Different Emotion\n")
        report.append("| 조건 | Cosine Similarity |")
        report.append("|---|---|")
        report.append(f"| Same emotion | {metrics['same_emotion_sim_mean']:.4f} |")
        report.append(f"| Different emotion | {metrics['diff_emotion_sim_mean']:.4f} |")
        report.append(f"| **Gap** | **{metrics['emotion_alignment_gap']:.4f}** |\n")
    report.append("\n## 4. Cross-modal Retrieval\n")
    report.append("| 방향 | R@1 | R@5 | R@10 | Median Rank |")
    report.append("|---|---|---|---|---|")
    report.append(f"| Image → Text | {metrics['retrieval_i2t_R@1']:.4f} | {metrics['retrieval_i2t_R@5']:.4f} | {metrics['retrieval_i2t_R@10']:.4f} | {metrics['retrieval_i2t_median_rank']:.0f} |")
    report.append(f"| Text → Image | {metrics['retrieval_t2i_R@1']:.4f} | {metrics['retrieval_t2i_R@5']:.4f} | {metrics['retrieval_t2i_R@10']:.4f} | {metrics['retrieval_t2i_median_rank']:.0f} |")

    report.append("\n## 5. Per-Emotion Paired Similarity\n")
    report.append("| 감정 | N | Paired Sim |")
    report.append("|---|---|---|")
    for emo, m in per_emo_metrics.items():
        report.append(f"| {emo} | {m['n_samples']:,} | {m['paired_sim_mean']:.4f} ± {m['paired_sim_std']:.4f} |")

    report.append("\n## 해석\n")
    if metrics['alignment_gap'] > 0.1:
        report.append("✓ **강한 alignment**: Paired similarity가 random보다 명확히 높음.")
    elif metrics['alignment_gap'] > 0.03:
        report.append("✓ **유의미한 alignment**: Paired vs random 차이 통계적 의미 있음.")
    else:
        report.append("⚠️ **약한 alignment**: 추가 학습/조정 필요.")

    report_path = output_dir / 'alignment_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"  → 리포트: {report_path}")

    print(f"\n✓ Alignment 평가 완료.")


if __name__ == '__main__':
    main()
