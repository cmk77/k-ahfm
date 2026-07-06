"""
C-block 2차 (B6v2): 환자 단위 그래프/하이퍼그래프 빌더.

[v2 변경사항]
원래 설계 (paper §3.4)를 *완성도 높여* 구현:
    - 5번째 hyperedge type 추가: E_affective_pattern
    - 노드 feature에 emotion + V/A 추가 (569-dim)
    - f_T의 8-class emotion + V/A regression 출력 활용

E_affective_pattern (4개 hyperedge):
    1. E_low_valence: 부정 정서 (우울 신호) - valence 평균 이하 세션
    2. E_high_arousal: 흥분/긴장 (불안 신호) - arousal 평균 이상 세션
    3. E_distress: 만성 디스트레스 (low valence + high arousal)
    4. E_emotional_volatility: 감정 분포 entropy 큰 세션 (변동성)

본 모델의 이론적 근거:
    Russell의 circumplex model + Plutchik의 wheel of emotions:
    - 우울증: 지속적 슬픔 → 낮은 valence
    - 불안장애: 두려움/걱정 → 높은 arousal + 낮은 valence
    - 중독: 감정 변동성 + 짜증/분노 → 변동성 큰 entropy

노드 features (569차원):
    - paragraph 임베딩 평균 (512)
    - paragraph 메타 평균 (5)
    - 28-증상 max-pool (28)
    - 11-개입 max-pool (11)
    - 3-차원 심각도 (3)
    - [신규] 8-class emotion softmax mean (8)  ← f_T classifier_emo 출력
    - [신규] valence + arousal normalized (2)   ← f_T classifier_av 출력
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# v1 vs v2 노드 feature 차원
NODE_FEATURE_DIM_V1 = 512 + 5 + 28 + 11 + 3                # 559 (기존 B4/B5/B6)
EMOTION_DIM = 8 + 2                                          # 8 emotion softmax + V/A
NODE_FEATURE_DIM_V2 = NODE_FEATURE_DIM_V1 + EMOTION_DIM       # 569 (B6v2)

# 기본값 (B4/B5/B6 호환성 위해 559 유지, B6v2만 569)
NODE_FEATURE_DIM = NODE_FEATURE_DIM_V1

SEVERITY_NORMALIZER = 6.0

# DSM-5 클러스터 indices (28-증상)
MDD_SYMPTOM_INDICES = list(range(0, 10))
GAD_SYMPTOM_INDICES = list(range(10, 17))
SUD_SYMPTOM_INDICES = list(range(17, 28))

# Edge types
EDGE_TYPE_TRAJECTORY = 0
EDGE_TYPE_CO_TRAJECTORY = 1
EDGE_TYPE_INTERVENTION_RESPONSE = 2
EDGE_TYPE_CLINICAL_PRIOR = 3
EDGE_TYPE_AFFECTIVE_PATTERN = 4                               # [신규]
NUM_EDGE_TYPES = 5                                            # 4 → 5

MAX_NODES = 11
MAX_HYPEREDGES = 25                                           # 17 → 25 (5번째 type 추가)


@dataclass
class PatientGraph:
    patient_id: str
    diagnosis: str
    num_nodes: int

    # (N, feature_dim) — feature_dim은 model_type에 따라 559 또는 569
    node_features: np.ndarray

    # B4용
    adjacency: np.ndarray

    # B5/B6/B6v2용
    incidence: np.ndarray
    num_hyperedges: int
    edge_types: np.ndarray

    # 환자 단위 features
    severity_first_3: np.ndarray
    severity_mean_input: np.ndarray

    # 라벨
    tr_label: int
    tr_valid: bool
    diag_label: int


# ============================================================================
# Emotion features 로드 헬퍼
# ============================================================================
class EmotionFeaturesLoader:
    """emotion_features.json 로드 및 세션별 조회."""

    def __init__(self, path: Optional[Path]):
        self.data = None
        if path is not None and Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            self.data = obj.get('sessions', obj)
            print(f"  emotion features 로드: {len(self.data):,} 세션")

    def get(self, patient_id: str, session_number: int) -> Optional[Dict]:
        if self.data is None:
            return None
        key = f"{patient_id}_s{session_number:02d}"
        return self.data.get(key)

    def to_session_vector(self, patient_id: str, session_number: int) -> np.ndarray:
        """단일 세션의 10-dim emotion feature 벡터 (8 emotion softmax mean + 2 V/A normalized mean)."""
        r = self.get(patient_id, session_number)
        if r is None:
            return np.zeros(EMOTION_DIM, dtype=np.float32)
        emo = np.array(r['emotion_softmax_mean'], dtype=np.float32)  # (8,)
        va = np.array([r['valence_norm_mean'], r['arousal_norm_mean']], dtype=np.float32)
        return np.concatenate([emo, va])


# ============================================================================
# 단일 세션 → 노드 feature
# ============================================================================
def extract_session_node_features(
    session,
    embedding: np.ndarray,
    max_paragraphs: int = 300,
    emotion_loader: Optional[EmotionFeaturesLoader] = None,
    include_emotion: bool = False,
) -> np.ndarray:
    """단일 세션 → 559 또는 569차원 노드 feature."""
    from data.normalize import session_full_features

    # paragraph 임베딩 평균
    if embedding.shape[0] == 0:
        para_emb_mean = np.zeros(512, dtype=np.float32)
    else:
        emb = embedding[:max_paragraphs] if embedding.shape[0] > max_paragraphs else embedding
        para_emb_mean = emb.mean(axis=0).astype(np.float32)

    # paragraph 메타 평균
    paragraphs = session.paragraphs[:max_paragraphs]
    if not paragraphs:
        para_meta_mean = np.zeros(5, dtype=np.float32)
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
    symptom_max = feats['symptoms_28'].astype(np.float32)
    intervention_max = feats['interventions_11'].astype(np.float32)

    severity = np.array([
        float(getattr(session, 'depression_severity', 0)) / SEVERITY_NORMALIZER,
        float(getattr(session, 'anxiety_severity', 0)) / SEVERITY_NORMALIZER,
        float(getattr(session, 'addiction_severity', 0)) / SEVERITY_NORMALIZER,
    ], dtype=np.float32)
    severity = np.clip(severity, 0.0, 1.0)

    # 기본 559차원
    base_feat = np.concatenate([
        para_emb_mean, para_meta_mean, symptom_max, intervention_max, severity,
    ]).astype(np.float32)

    if not include_emotion:
        return base_feat

    # 569차원 (v2): + emotion features
    if emotion_loader is not None:
        emotion_vec = emotion_loader.to_session_vector(session.patient_id, session.session_number)
    else:
        emotion_vec = np.zeros(EMOTION_DIM, dtype=np.float32)

    return np.concatenate([base_feat, emotion_vec]).astype(np.float32)


def load_session_embedding(session, embeddings_cache_dir: Path) -> np.ndarray:
    emb_path = embeddings_cache_dir / f"{session.patient_id}_s{session.session_number:02d}_para_emb.npy"
    if not emb_path.exists():
        return np.zeros((0, 512), dtype=np.float32)
    return np.load(emb_path).astype(np.float32)


# ============================================================================
# B4: Temporal Graph
# ============================================================================
def build_temporal_adjacency(num_nodes: int) -> np.ndarray:
    A = np.eye(num_nodes, dtype=np.float32)
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


# ============================================================================
# B5: Single Hyperedge
# ============================================================================
def build_single_hyperedge_incidence(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    H = np.ones((num_nodes, 1), dtype=np.float32)
    edge_types = np.array([EDGE_TYPE_TRAJECTORY], dtype=np.int64)
    return H, edge_types


# ============================================================================
# B6: K-AHFM-Clinic 4-Hyperedge Hypergraph (v1 — 기존)
# ============================================================================
def build_kahfm_hypergraph(
    node_features: np.ndarray,
    diagnosis: str,
    threshold_symptom_activity: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """B6 v1: 4종 hyperedge (기존)."""
    N = node_features.shape[0]
    # symptom_per_session: indices 512+5 ~ 512+5+28 in feature
    symptom_per_session = node_features[:, 517:545]
    intervention_per_session = node_features[:, 545:556]

    edges_list = []
    edge_types_list = []

    # 1. E_trajectory
    edges_list.append(np.ones(N, dtype=np.float32))
    edge_types_list.append(EDGE_TYPE_TRAJECTORY)

    # 2. E_co-trajectory (high vs low 증상 강도)
    session_symptom_intensity = symptom_per_session.sum(axis=1)
    threshold = np.median(session_symptom_intensity)
    high_mask = (session_symptom_intensity >= threshold).astype(np.float32)
    low_mask = (session_symptom_intensity < threshold).astype(np.float32)
    if high_mask.sum() >= 1:
        edges_list.append(high_mask)
        edge_types_list.append(EDGE_TYPE_CO_TRAJECTORY)
    if low_mask.sum() >= 1:
        edges_list.append(low_mask)
        edge_types_list.append(EDGE_TYPE_CO_TRAJECTORY)

    # 3. E_intervention-response
    for int_idx in range(11):
        int_mask = (intervention_per_session[:, int_idx] > threshold_symptom_activity).astype(np.float32)
        if int_mask.sum() >= 1:
            edges_list.append(int_mask)
            edge_types_list.append(EDGE_TYPE_INTERVENTION_RESPONSE)

    # 4. E_clinical_prior
    for cluster_indices in [MDD_SYMPTOM_INDICES, GAD_SYMPTOM_INDICES, SUD_SYMPTOM_INDICES]:
        cluster_activity = symptom_per_session[:, cluster_indices].sum(axis=1)
        cluster_threshold = max(cluster_activity.mean(), threshold_symptom_activity)
        cluster_mask = (cluster_activity >= cluster_threshold).astype(np.float32)
        if cluster_mask.sum() >= 1:
            edges_list.append(cluster_mask)
            edge_types_list.append(EDGE_TYPE_CLINICAL_PRIOR)

    if not edges_list:
        H = np.ones((N, 1), dtype=np.float32)
        edge_types = np.array([EDGE_TYPE_TRAJECTORY], dtype=np.int64)
    else:
        H = np.stack(edges_list, axis=1).astype(np.float32)
        edge_types = np.array(edge_types_list, dtype=np.int64)
    return H, edge_types


# ============================================================================
# B6v2: K-AHFM-Clinic 5-Hyperedge (신규 — 5th type 추가)
# ============================================================================
def build_kahfm_v2_hypergraph(
    node_features_v2: np.ndarray,            # (N, 569)
    diagnosis: str,
    threshold_symptom_activity: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """B6v2: 4종 hyperedge + 5th type E_affective_pattern.

    node_features_v2: 569-dim (마지막 10개가 emotion softmax mean + V/A normalized)
    """
    N = node_features_v2.shape[0]

    # 기존 4종 hyperedge는 v1 로직 그대로 사용 (첫 559차원만 활용)
    H_base, edge_types_base = build_kahfm_hypergraph(
        node_features_v2[:, :NODE_FEATURE_DIM_V1], diagnosis, threshold_symptom_activity,
    )

    # 5번째 type: E_affective_pattern
    # node features 마지막 10개: [emo_0,...,emo_7, valence, arousal]
    emotion_softmax = node_features_v2[:, -EMOTION_DIM:-2]    # (N, 8)
    valence = node_features_v2[:, -2]                          # (N,)
    arousal = node_features_v2[:, -1]                          # (N,)

    affective_edges = []

    # 1. E_low_valence: 낮은 valence (우울 신호)
    v_threshold = np.median(valence)
    low_v_mask = (valence <= v_threshold).astype(np.float32)
    if low_v_mask.sum() >= 1:
        affective_edges.append(low_v_mask)

    # 2. E_high_arousal: 높은 arousal (불안 신호)
    a_threshold = np.median(arousal)
    high_a_mask = (arousal > a_threshold).astype(np.float32)
    if high_a_mask.sum() >= 1:
        affective_edges.append(high_a_mask)

    # 3. E_distress: low valence + high arousal (만성 디스트레스)
    distress_mask = ((valence <= v_threshold) & (arousal > a_threshold)).astype(np.float32)
    if distress_mask.sum() >= 1:
        affective_edges.append(distress_mask)

    # 4. E_emotional_volatility: 감정 분포 entropy 큰 세션 (변동성)
    entropy_per_session = -(emotion_softmax * np.log(emotion_softmax + 1e-8)).sum(axis=1)
    e_threshold = np.median(entropy_per_session)
    volatility_mask = (entropy_per_session > e_threshold).astype(np.float32)
    if volatility_mask.sum() >= 1:
        affective_edges.append(volatility_mask)

    # 통합
    if affective_edges:
        H_affective = np.stack(affective_edges, axis=1).astype(np.float32)
        edge_types_affective = np.full(H_affective.shape[1], EDGE_TYPE_AFFECTIVE_PATTERN, dtype=np.int64)
        H = np.concatenate([H_base, H_affective], axis=1)
        edge_types = np.concatenate([edge_types_base, edge_types_affective])
    else:
        H = H_base
        edge_types = edge_types_base

    return H, edge_types


# ============================================================================
# 환자 단위 그래프 빌드
# ============================================================================
def build_patient_graph(
    patient_id: str,
    diagnosis: str,
    sessions: List,
    embeddings_cache_dir: Path,
    patient_label,
    model_type: str,                              # 'B4' | 'B5' | 'B6' | 'B6v2'
    max_paragraphs: int = 300,
    emotion_loader: Optional[EmotionFeaturesLoader] = None,
) -> Optional[PatientGraph]:
    if not sessions:
        return None

    sessions_sorted = sorted(sessions, key=lambda s: s.session_number)
    N = min(len(sessions_sorted), MAX_NODES)
    sessions_sorted = sessions_sorted[:N]

    # v2 모델은 emotion features 포함, 그 외는 v1
    include_emotion = (model_type == 'B6v2')
    feat_dim = NODE_FEATURE_DIM_V2 if include_emotion else NODE_FEATURE_DIM_V1

    node_features = np.zeros((N, feat_dim), dtype=np.float32)
    for i, s in enumerate(sessions_sorted):
        emb = load_session_embedding(s, embeddings_cache_dir)
        node_features[i] = extract_session_node_features(
            s, emb, max_paragraphs,
            emotion_loader=emotion_loader,
            include_emotion=include_emotion,
        )

    severity_first_3 = node_features[0, 556:559].copy()
    severity_mean_input = node_features[:, 556:559].mean(axis=0)

    if model_type == 'B4':
        adjacency = build_temporal_adjacency(N)
        incidence = np.zeros((N, 1), dtype=np.float32)
        edge_types = np.array([0], dtype=np.int64)
        num_hyperedges = 0
    elif model_type == 'B5':
        adjacency = np.zeros((N, N), dtype=np.float32)
        incidence, edge_types = build_single_hyperedge_incidence(N)
        num_hyperedges = 1
    elif model_type == 'B6':
        adjacency = np.zeros((N, N), dtype=np.float32)
        incidence, edge_types = build_kahfm_hypergraph(node_features, diagnosis)
        num_hyperedges = incidence.shape[1]
    elif model_type == 'B6v2':
        adjacency = np.zeros((N, N), dtype=np.float32)
        incidence, edge_types = build_kahfm_v2_hypergraph(node_features, diagnosis)
        num_hyperedges = incidence.shape[1]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    tr_label = patient_label.tr_label if patient_label.tr_valid else -1
    diag_label = patient_label.diagnosis_idx

    return PatientGraph(
        patient_id=patient_id, diagnosis=diagnosis,
        num_nodes=N, node_features=node_features,
        adjacency=adjacency,
        incidence=incidence, num_hyperedges=num_hyperedges, edge_types=edge_types,
        severity_first_3=severity_first_3,
        severity_mean_input=severity_mean_input,
        tr_label=tr_label, tr_valid=patient_label.tr_valid,
        diag_label=diag_label,
    )


def build_all_patient_graphs(
    sessions_by_patient: Dict[str, List],
    patient_labels: Dict,
    embeddings_cache_dir: Path,
    model_type: str,
    scenario: str = 'early',
    input_window: int = 3,
    max_paragraphs: int = 300,
    progress: bool = True,
    emotion_features_path: Optional[Path] = None,
) -> Dict[str, PatientGraph]:
    """전체 환자 그래프 일괄 빌드. B6v2의 경우 emotion_features_path 필요."""
    if model_type == 'B6v2' and emotion_features_path is None:
        raise ValueError("B6v2 requires emotion_features_path (emotion_features.json)")

    emotion_loader = EmotionFeaturesLoader(emotion_features_path) if emotion_features_path else None

    graphs: Dict[str, PatientGraph] = {}
    iterator = sessions_by_patient.items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(list(iterator), desc=f'{model_type} 그래프 ({scenario})', unit='환자')

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

        g = build_patient_graph(
            patient_id=pid, diagnosis=label.diagnosis,
            sessions=selected, embeddings_cache_dir=embeddings_cache_dir,
            patient_label=label, model_type=model_type,
            max_paragraphs=max_paragraphs,
            emotion_loader=emotion_loader,
        )
        if g is not None:
            graphs[pid] = g

    return graphs


# ============================================================================
# 배치 padding
# ============================================================================
def pad_and_batch_graphs(
    graphs: List[PatientGraph],
    model_type: str,
) -> Dict:
    import torch

    B = len(graphs)
    max_N = max(g.num_nodes for g in graphs)
    max_E = max(g.num_hyperedges if g.num_hyperedges > 0 else 1 for g in graphs)
    # 노드 차원 (B6v2는 569, 나머지는 559)
    feat_dim = graphs[0].node_features.shape[1]

    node_features = torch.zeros(B, max_N, feat_dim, dtype=torch.float32)
    node_mask = torch.zeros(B, max_N, dtype=torch.bool)

    adjacency = torch.zeros(B, max_N, max_N, dtype=torch.float32) if model_type == 'B4' else None
    incidence = torch.zeros(B, max_N, max_E, dtype=torch.float32) if model_type in ('B5', 'B6', 'B6v2') else None
    edge_mask = torch.zeros(B, max_E, dtype=torch.bool) if model_type in ('B5', 'B6', 'B6v2') else None
    edge_types = torch.zeros(B, max_E, dtype=torch.long) if model_type in ('B6', 'B6v2') else None

    severity_first_3 = torch.zeros(B, 3, dtype=torch.float32)
    severity_mean_input = torch.zeros(B, 3, dtype=torch.float32)
    tr_label = torch.zeros(B, dtype=torch.long)
    tr_valid = torch.zeros(B, dtype=torch.long)
    diag_label = torch.zeros(B, dtype=torch.long)
    patient_ids, diagnoses = [], []

    for b, g in enumerate(graphs):
        N, E = g.num_nodes, g.num_hyperedges
        node_features[b, :N] = torch.from_numpy(g.node_features)
        node_mask[b, :N] = True

        if model_type == 'B4':
            adjacency[b, :N, :N] = torch.from_numpy(g.adjacency)
        elif model_type in ('B5', 'B6', 'B6v2'):
            E_use = max(E, 1)
            incidence[b, :N, :E_use] = torch.from_numpy(g.incidence[:, :E_use])
            edge_mask[b, :E_use] = True
            if model_type in ('B6', 'B6v2'):
                edge_types[b, :E_use] = torch.from_numpy(g.edge_types[:E_use])

        severity_first_3[b] = torch.from_numpy(g.severity_first_3)
        severity_mean_input[b] = torch.from_numpy(g.severity_mean_input)
        tr_label[b] = max(g.tr_label, 0)
        tr_valid[b] = 1 if g.tr_valid else 0
        diag_label[b] = g.diag_label
        patient_ids.append(g.patient_id)
        diagnoses.append(g.diagnosis)

    out = {
        'node_features': node_features,
        'node_mask': node_mask,
        'severity_first_3': severity_first_3,
        'severity_mean_input': severity_mean_input,
        'tr_label': tr_label, 'tr_valid': tr_valid, 'diag_label': diag_label,
        'patient_id': patient_ids, 'diagnosis': diagnoses,
    }
    if adjacency is not None:
        out['adjacency'] = adjacency
    if incidence is not None:
        out['incidence'] = incidence
        out['edge_mask'] = edge_mask
    if edge_types is not None:
        out['edge_types'] = edge_types
    return out
