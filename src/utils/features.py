"""
C-block 1차 (v3): 시계열 보존 + 임상가 심각도 입력.

[v3 변경사항 vs v2]
v2는 시계열 보존 (mean/delta)을 추가했지만 결정적 한계가 있었음:
  - TR 라벨이 *임상가 평가 심각도* 변화로 정의되는데,
  - 모델 입력은 *텍스트 임베딩* 변화만 갖고 있음
  - 두 차원의 상관관계가 약해 라벨에 직접 정합하지 못함

v3는 *임상가 평가 심각도*를 직접 입력으로 추가:
  - severity_first_3 (3차원): 첫 회기 depression/anxiety/addiction 심각도
  - severity_mean_input (3차원): 입력 회기들의 평균 심각도
  - 합계 +6 차원만 추가 (작지만 신호가 강함)

라벨 누설 검증:
  - severity_first는 첫 회기 정보 → S_early의 부분, S_late는 모름 (누설 아님)
  - severity_mean_input은 입력 회기 평균 → Early=S_early, Full=평균 (누설 아님)
  - last block severity는 *입력하지 않음* (Full scenario에서 직접 누설이 될 수 있음)

새 차원:
    B1 = 1024 → 1030 (+6)
    B2 = 1034 → 1040 (+6)
    B3 = 1090 → 1096 (+6)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


EMBEDDING_DIM = 512
PARAGRAPH_META_DIM = 5
SESSION_SYMPTOM_DIM = 28
SEVERITY_DIM = 3                      # depression, anxiety, addiction 0-3 (정규화 후 0-1)

# v3: + severity_first_3 + severity_mean_input
B1_FEATURE_DIM = 2 * EMBEDDING_DIM + 2 * SEVERITY_DIM                              # 1030
B2_FEATURE_DIM = 2 * EMBEDDING_DIM + 2 * PARAGRAPH_META_DIM + 2 * SEVERITY_DIM     # 1040
B3_FEATURE_DIM = (2 * EMBEDDING_DIM + 2 * PARAGRAPH_META_DIM
                  + 2 * SESSION_SYMPTOM_DIM + 2 * SEVERITY_DIM)                     # 1096


@dataclass
class PatientFeature:
    patient_id: str
    diagnosis: str
    b1_feature: np.ndarray  # (1030,)
    b2_feature: np.ndarray  # (1040,)
    b3_feature: np.ndarray  # (1096,)
    tr_label: int
    tr_valid: bool
    diag_label: int
    n_observed_sessions: int


def extract_session_features(
    session,
    embedding: np.ndarray,
    max_paragraphs: int = 300,
) -> Dict[str, np.ndarray]:
    """단일 세션 features (paragraph emb, meta, 28-증상 max, severity)."""
    from data.normalize import session_full_features

    if embedding.shape[0] == 0:
        para_emb_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    else:
        emb = embedding[:max_paragraphs] if embedding.shape[0] > max_paragraphs else embedding
        para_emb_mean = emb.mean(axis=0).astype(np.float32)

    paragraphs = session.paragraphs[:max_paragraphs]
    if not paragraphs:
        para_meta_mean = np.zeros(PARAGRAPH_META_DIM, dtype=np.float32)
    else:
        metas = []
        for p in paragraphs:
            duration = max(p.end_point - p.start_point, 0.0) if p.start_point is not None else 0.0
            char_count = len(p.text) if p.text else 0
            cps = char_count / max(duration, 0.1) if duration > 0 else 0.0
            log_cc = np.log1p(char_count)
            speaker_counselor = 1.0 if p.speaker in ('상담사', 'counselor', 'C') else 0.0
            speaker_client = 1.0 if p.speaker in ('내담자', 'client', 'P', 'patient') else 0.0
            metas.append([cps, log_cc, duration, speaker_counselor, speaker_client])
        para_meta_mean = np.mean(metas, axis=0).astype(np.float32)

    feats = session_full_features(session)

    # v3 신규: 임상가 평가 심각도 (depression, anxiety, addiction) 0-3 → 0-1 정규화
    severity = np.array([
        float(getattr(session, 'depression_severity', 0)) / 3.0,
        float(getattr(session, 'anxiety_severity', 0)) / 3.0,
        float(getattr(session, 'addiction_severity', 0)) / 3.0,
    ], dtype=np.float32)
    # 클리핑 (0-1 범위 보장)
    severity = np.clip(severity, 0.0, 1.0)

    return {
        'para_emb_mean': para_emb_mean,
        'para_meta_mean': para_meta_mean,
        'symptom_max': feats['symptoms_28'].astype(np.float32),
        'severity': severity,
    }


def aggregate_block_features(
    sessions: List,
    embeddings_cache_dir: Path,
    max_paragraphs: int = 300,
) -> Dict[str, np.ndarray]:
    """세션 블록 features 평균 집계."""
    if not sessions:
        return {
            'para_emb': np.zeros(EMBEDDING_DIM, dtype=np.float32),
            'para_meta': np.zeros(PARAGRAPH_META_DIM, dtype=np.float32),
            'symptom': np.zeros(SESSION_SYMPTOM_DIM, dtype=np.float32),
            'severity': np.zeros(SEVERITY_DIM, dtype=np.float32),
        }

    session_features = []
    for s in sessions:
        emb_path = embeddings_cache_dir / f"{s.patient_id}_s{s.session_number:02d}_para_emb.npy"
        if not emb_path.exists():
            embedding = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        else:
            embedding = np.load(emb_path).astype(np.float32)

        sf = extract_session_features(s, embedding, max_paragraphs=max_paragraphs)
        session_features.append(sf)

    return {
        'para_emb': np.mean([sf['para_emb_mean'] for sf in session_features], axis=0).astype(np.float32),
        'para_meta': np.mean([sf['para_meta_mean'] for sf in session_features], axis=0).astype(np.float32),
        'symptom': np.mean([sf['symptom_max'] for sf in session_features], axis=0).astype(np.float32),
        'severity': np.mean([sf['severity'] for sf in session_features], axis=0).astype(np.float32),
    }


def split_into_blocks(sessions: List) -> Tuple[List, List]:
    """라벨 정의(S_early/S_late)와 정합하는 first/last 블록 분리."""
    n = len(sessions)
    if n >= 6:
        return sessions[:3], sessions[-3:]
    if n in (4, 5):
        half = n // 2
        return sessions[:half], sessions[-half:]
    if n == 3:
        return sessions[:1], sessions[-1:]
    if n == 2:
        return sessions[:1], sessions[-1:]
    return sessions, sessions


def extract_patient_feature(
    patient_id: str,
    diagnosis: str,
    sessions: List,
    embeddings_cache_dir: Path,
    patient_label,
    max_paragraphs: int = 300,
) -> Optional[PatientFeature]:
    """환자 단위 feature 추출 — v3 시계열 보존 + 임상 심각도."""
    if not sessions:
        return None

    sessions_sorted = sorted(sessions, key=lambda s: s.session_number)
    first_block, last_block = split_into_blocks(sessions_sorted)

    first_feat = aggregate_block_features(first_block, embeddings_cache_dir, max_paragraphs)
    last_feat = aggregate_block_features(last_block, embeddings_cache_dir, max_paragraphs)
    all_feat = aggregate_block_features(sessions_sorted, embeddings_cache_dir, max_paragraphs)

    # Delta 계산
    delta_emb = (last_feat['para_emb'] - first_feat['para_emb']).astype(np.float32)
    delta_meta = (last_feat['para_meta'] - first_feat['para_meta']).astype(np.float32)
    delta_symptom = (last_feat['symptom'] - first_feat['symptom']).astype(np.float32)

    # v3 신규: 임상 심각도
    # severity_first_3: 첫 회기 (또는 첫 블록의 평균) 심각도
    severity_first = first_feat['severity'].astype(np.float32)
    # severity_mean_input: 입력 회기들의 평균 심각도
    severity_mean = all_feat['severity'].astype(np.float32)

    # B1: paragraph emb 만 + severity
    b1 = np.concatenate([
        all_feat['para_emb'], delta_emb,
        severity_first, severity_mean,
    ]).astype(np.float32)

    # B2: + paragraph meta
    b2 = np.concatenate([
        all_feat['para_emb'], delta_emb,
        all_feat['para_meta'], delta_meta,
        severity_first, severity_mean,
    ]).astype(np.float32)

    # B3: + 28-symptom
    b3 = np.concatenate([
        all_feat['para_emb'], delta_emb,
        all_feat['para_meta'], delta_meta,
        all_feat['symptom'], delta_symptom,
        severity_first, severity_mean,
    ]).astype(np.float32)

    tr_label = patient_label.tr_label if patient_label.tr_valid else -1
    diag_label = patient_label.diagnosis_idx

    return PatientFeature(
        patient_id=patient_id, diagnosis=diagnosis,
        b1_feature=b1, b2_feature=b2, b3_feature=b3,
        tr_label=tr_label, tr_valid=patient_label.tr_valid,
        diag_label=diag_label,
        n_observed_sessions=len(sessions_sorted),
    )


def extract_all_patient_features(
    sessions_by_patient: Dict[str, List],
    patient_labels: Dict,
    embeddings_cache_dir: Path,
    scenario: str = 'early',
    input_window: int = 3,
    max_paragraphs: int = 300,
    progress: bool = True,
) -> Dict[str, PatientFeature]:
    """전체 환자 baseline feature 일괄 추출 (v3)."""
    features: Dict[str, PatientFeature] = {}

    iterator = sessions_by_patient.items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(list(iterator), desc=f'feature 추출 ({scenario})', unit='환자')

    for pid, sess_list in iterator:
        if pid not in patient_labels:
            continue
        label = patient_labels[pid]
        sess_sorted = sorted(sess_list, key=lambda s: s.session_number)

        if scenario == 'early':
            selected = sess_sorted[:input_window]
        elif scenario == 'full':
            selected = sess_sorted
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        if not selected:
            continue

        feat = extract_patient_feature(
            patient_id=pid, diagnosis=label.diagnosis,
            sessions=selected, embeddings_cache_dir=embeddings_cache_dir,
            patient_label=label, max_paragraphs=max_paragraphs,
        )
        if feat is not None:
            features[pid] = feat

    return features
