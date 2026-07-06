"""
M1: 한국어 텍스트 감정 인코더 f_T.

구조: KLUE/RoBERTa-base backbone + multi-task head
    - 8-class emotion classifier (primary: multimodal)
    - 8-class emotion classifier (auxiliary: text-only)
    - Arousal regressor
    - Valence regressor
    - 512-dim emotion embedding (downstream에서 사용)

학습 후 KLUE 백본 + projection head를 저장하여 K-AHFM-Clinic의 Stage 1에서
#58 paragraph 임베딩 생성에 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================================
# 모델 설정
# ============================================================================
@dataclass
class TextEncoderConfig:
    backbone_name: str = 'klue/roberta-base'
    num_emotion_classes: int = 8
    embedding_dim: int = 512           # 다운스트림 사용 임베딩 차원
    dropout: float = 0.1
    pooling: str = 'cls'               # 'cls' 또는 'mean'


# ============================================================================
# 모델
# ============================================================================
class KoreanTextEmotionEncoder(nn.Module):
    """KLUE/RoBERTa + multi-task head.

    forward 반환:
        emotion_embedding: (B, embedding_dim)  ← 다운스트림 K-AHFM-Clinic 사용
        emotion_mm_logits: (B, num_emotion_classes)
        emotion_text_logits: (B, num_emotion_classes)
        valence_pred: (B,)
        arousal_pred: (B,)
    """

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.backbone_name)
        hidden_size = self.backbone.config.hidden_size  # KLUE-base: 768

        # 다운스트림용 emotion embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_size, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.Dropout(config.dropout),
        )

        # 주 분류기: emotion.multimodal
        self.emo_mm_head = nn.Linear(config.embedding_dim, config.num_emotion_classes)

        # 보조 분류기: emotion.text
        self.emo_text_head = nn.Linear(config.embedding_dim, config.num_emotion_classes)

        # 회귀 헤드: valence, arousal (각각 sigmoid로 [0, 1])
        self.av_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 2),  # [valence, arousal]
        )

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """토큰 hidden states → 단일 벡터."""
        if self.config.pooling == 'cls':
            return hidden_states[:, 0, :]
        elif self.config.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        # Backbone forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self._pool(outputs.last_hidden_state, attention_mask)

        # 임베딩
        emb = self.embedding_proj(pooled)  # (B, embedding_dim)

        # Heads
        emo_mm_logits = self.emo_mm_head(emb)
        emo_text_logits = self.emo_text_head(emb)
        av = torch.sigmoid(self.av_head(emb))  # (B, 2), [0, 1]
        valence = av[:, 0]
        arousal = av[:, 1]

        return {
            'emotion_embedding': emb,
            'emotion_mm_logits': emo_mm_logits,
            'emotion_text_logits': emo_text_logits,
            'valence_pred': valence,
            'arousal_pred': arousal,
        }

    def extract_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """추론 전용: 임베딩만 반환 (다운스트림 사용용)."""
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pooled = self._pool(outputs.last_hidden_state, attention_mask)
            emb = self.embedding_proj(pooled)
        return emb


# ============================================================================
# Loss 함수
# ============================================================================
def multitask_loss(
    outputs: dict,
    batch: dict,
    lambda_mm: float = 1.0,
    lambda_text: float = 0.3,
    lambda_av: float = 0.3,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """다중 과제 loss 계산.

    Args:
        outputs: 모델 forward 결과 dict
        batch: dataloader batch dict
        lambda_*: 각 loss 가중치
        class_weights: (num_classes,) 클래스 균형 가중 (선택)

    Returns:
        total_loss, loss_dict (각 sub-loss 값)
    """
    # 주 task: multimodal emotion
    loss_mm = F.cross_entropy(
        outputs['emotion_mm_logits'],
        batch['emotion_mm_label'],
        weight=class_weights,
        ignore_index=-100,
    )

    # 보조 task 1: text emotion
    loss_text = F.cross_entropy(
        outputs['emotion_text_logits'],
        batch['emotion_text_label'],
        weight=class_weights,
        ignore_index=-100,
    )

    # 보조 task 2: arousal/valence MSE
    loss_v = F.mse_loss(outputs['valence_pred'], batch['valence'])
    loss_a = F.mse_loss(outputs['arousal_pred'], batch['arousal'])
    loss_av = loss_v + loss_a

    total = lambda_mm * loss_mm + lambda_text * loss_text + lambda_av * loss_av

    return total, {
        'total': total.item(),
        'loss_mm': loss_mm.item(),
        'loss_text': loss_text.item() if not torch.isnan(loss_text) else 0.0,
        'loss_valence': loss_v.item(),
        'loss_arousal': loss_a.item(),
    }
