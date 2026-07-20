#!/usr/bin/env python
"""K-AHFM-Clinic Docker 데모.

AI Hub #58 심리상담 데이터는 라이선스상 재배포가 불가능하므로, 이 데모는
두 부분으로 구성된다:

  [1부] 합성(synthetic) 코호트로 K-AHFM-Clinic 본 모델(B6)의 전체 학습
        파이프라인을 실행하는 스모크 테스트
        — 환자별 세션 하이퍼그래프 구성 (4종 hyperedge)
        — adaptive hyperedge weighting + 멀티태스크 학습
          (주 과제: 치료반응 이진분류 / 보조 과제: 4-class 진단분류)
        — early-prediction 시나리오 재현: 입력은 첫 3회기뿐, 라벨은
          전체 경과(첫/마지막 3회기 심각도 평균 차)로 정의
        ※ 합성 데이터이므로 이 수치 자체는 연구 결과가 아니다.

  [2부] 리포지토리에 저장된 실제 논문 실험 결과 요약 출력
        (실측 실험: 환자 186명 / 세션 1,313개 / 5-fold × 5-seed)

실행: docker run --rm ghcr.io/cmk77/k-ahfm:latest
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from graph.build_graph import (
    NODE_FEATURE_DIM_V1,
    MDD_SYMPTOM_INDICES,
    GAD_SYMPTOM_INDICES,
    SUD_SYMPTOM_INDICES,
    PatientGraph,
    build_kahfm_hypergraph,
    pad_and_batch_graphs,
)
from models.graph_models import build_graph_model, multitask_loss

SEED = 42
RESPONSE_THRESHOLD = 1.0     # 실제 파이프라인(src/data/labels.py)과 동일
EARLY_WINDOW = 3             # early 시나리오: 첫 3회기만 입력
SEVERITY_NORMALIZER = 6.0    # build_graph.py와 동일

DIAG_NAMES = ['depression', 'anxiety', 'addiction', 'normal']
DIAG_CLUSTER = {
    0: MDD_SYMPTOM_INDICES,
    1: GAD_SYMPTOM_INDICES,
    2: SUD_SYMPTOM_INDICES,
}
EDGE_TYPE_NAMES = {
    0: 'E_trajectory',
    1: 'E_co-trajectory',
    2: 'E_intervention',
    3: 'E_clinical_prior',
    4: 'E_affective',
}

# 합성 임베딩 공간: 진단군별 고정 중심 + 심각도 연동 방향 (재현성 위해 시드 고정)
_DIAG_EMB_CENTERS = {
    d: np.random.default_rng(1000 + d).standard_normal(512) * 0.6 for d in range(4)
}
_SEVERITY_TREND_DIR = np.random.default_rng(2000).standard_normal(512) * 0.4


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ============================================================================
# [1부] 합성 코호트 생성
# ============================================================================
def make_synthetic_patient(pid: int, rng: np.random.Generator):
    """실제 데이터 스키마(559-dim 노드, 0-3 심각도 궤적)를 모사한 합성 환자."""
    is_normal = rng.random() < 0.15
    diag = 3 if is_normal else int(rng.integers(0, 3))
    n_sess = int(rng.integers(6, 11))

    # 심각도 궤적 (0-3): responder는 회기에 걸쳐 감소, stable은 정체
    if is_normal:
        sev_traj = np.clip(rng.normal(0.2, 0.1, n_sess), 0, 3)
    else:
        planted_responder = rng.random() < 0.35
        s0 = rng.uniform(2.0, 3.0)
        rate = rng.uniform(0.18, 0.30) if planted_responder else rng.uniform(-0.02, 0.04)
        sev_traj = np.clip(
            s0 - rate * np.arange(n_sess) + rng.normal(0, 0.15, n_sess), 0, 3,
        )

    # TR 라벨은 실제 파이프라인과 동일하게 '전체 경과'에서 계산
    delta = float(np.mean(sev_traj[:3]) - np.mean(sev_traj[-3:]))
    if is_normal:
        tr_valid, tr_label = False, -1
    else:
        tr_valid = True
        tr_label = 1 if delta >= RESPONSE_THRESHOLD else 0

    # 노드 feature: 입력은 첫 EARLY_WINDOW 회기만 (early-prediction 시나리오)
    diag_emb_center = _DIAG_EMB_CENTERS[diag]              # 진단군별 고정 중심
    trend_dir = _SEVERITY_TREND_DIR                        # 심각도 연동 방향 벡터

    nodes = []
    for t in range(EARLY_WINDOW):
        sev_t = sev_traj[t]
        emb = diag_emb_center + trend_dir * (sev_t / 3.0) + rng.standard_normal(512) * 0.8

        meta = np.array([
            rng.uniform(2, 8),            # cps
            rng.uniform(2, 5),            # log(char count)
            rng.uniform(5, 60),           # duration
            rng.uniform(0.4, 0.6),        # counselor 비율
            rng.uniform(0.4, 0.6),        # client 비율
        ])

        symptoms = np.abs(rng.normal(0.03, 0.03, 28))
        if diag in DIAG_CLUSTER:
            idx = DIAG_CLUSTER[diag]
            symptoms[idx] = np.clip(sev_t / 3.0 + rng.normal(0, 0.1, len(idx)), 0, 1)

        interventions = (rng.random(11) < 0.3).astype(float) * rng.uniform(0.2, 1.0, 11)

        severity3 = np.zeros(3)
        if diag in DIAG_CLUSTER:
            severity3[diag] = sev_t
        severity3 = np.clip(severity3 / SEVERITY_NORMALIZER, 0, 1)

        nodes.append(np.concatenate([emb, meta, symptoms, interventions, severity3]))

    node_features = np.stack(nodes).astype(np.float32)
    assert node_features.shape[1] == NODE_FEATURE_DIM_V1

    diag_name = DIAG_NAMES[diag]
    incidence, edge_types = build_kahfm_hypergraph(node_features, diag_name)

    return PatientGraph(
        patient_id=f"SYN{pid:03d}",
        diagnosis=diag_name,
        num_nodes=node_features.shape[0],
        node_features=node_features,
        adjacency=np.zeros((EARLY_WINDOW, EARLY_WINDOW), dtype=np.float32),
        incidence=incidence,
        num_hyperedges=incidence.shape[1],
        edge_types=edge_types,
        severity_first_3=node_features[0, 556:559].copy(),
        severity_mean_input=node_features[:, 556:559].mean(axis=0),
        tr_label=tr_label,
        tr_valid=tr_valid,
        diag_label=diag,
    )


def compute_auroc(y_true, scores) -> float:
    """rank 기반 AUROC (외부 의존성 없이 계산)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def run_smoke_test() -> None:
    banner("[1부] 합성 코호트 스모크 테스트 — B6 (K-AHFM-Clinic)")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    n_patients = 90
    graphs = [make_synthetic_patient(i, rng) for i in range(n_patients)]

    n_tr_valid = sum(g.tr_valid for g in graphs)
    n_resp = sum(1 for g in graphs if g.tr_valid and g.tr_label == 1)
    print(f"\n합성 환자 {n_patients}명 생성 "
          f"(TR-valid {n_tr_valid}명, responder {n_resp}명, "
          f"normal {sum(1 for g in graphs if g.diagnosis == 'normal')}명)")
    print(f"입력: 환자당 첫 {EARLY_WINDOW}회기 → 세션 노드 {EARLY_WINDOW}개 × 559-dim")

    # 하이퍼그래프 구성 통계
    type_counts = {name: 0 for name in EDGE_TYPE_NAMES.values()}
    for g in graphs:
        for t in g.edge_types:
            type_counts[EDGE_TYPE_NAMES[int(t)]] += 1
    avg_edges = np.mean([g.num_hyperedges for g in graphs])
    print(f"하이퍼그래프: 환자당 평균 {avg_edges:.1f}개 hyperedge")
    for name, cnt in type_counts.items():
        if cnt:
            print(f"    {name:<18s}: 총 {cnt}개")

    # 환자 단위 train/test 분리
    perm = rng.permutation(n_patients)
    n_train = int(n_patients * 0.7)
    train_graphs = [graphs[i] for i in perm[:n_train]]
    test_graphs = [graphs[i] for i in perm[n_train:]]

    train_batch = pad_and_batch_graphs(train_graphs, 'B6')
    test_batch = pad_and_batch_graphs(test_graphs, 'B6')

    model = build_graph_model('B6', hidden_dim=64, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n모델: KAHFMClinic (adaptive 4-type hyperedge attention), "
          f"파라미터 {n_params:,}개, device=CPU")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    t0 = time.time()
    print(f"\n학습 (train 환자 {len(train_graphs)}명, full-batch, 120 epochs):")
    for epoch in range(1, 121):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_batch)
        loss, parts = multitask_loss(outputs, train_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 30 == 0:
            print(f"  epoch {epoch:3d}: total={parts['total']:.4f} "
                  f"(TR={parts['loss_tr']:.4f}, diag={parts['loss_diag']:.4f})")

    # 평가 (test 환자)
    model.eval()
    with torch.no_grad():
        out = model(test_batch, return_attention=True)
        tr_prob = torch.sigmoid(out['tr_logit']).numpy()
        tr_valid = test_batch['tr_valid'].bool().numpy()
        tr_label = test_batch['tr_label'].numpy()
        diag_pred = out['diag_logits'].argmax(-1).numpy()
        diag_label = test_batch['diag_label'].numpy()

    auroc = compute_auroc(tr_label[tr_valid], tr_prob[tr_valid])
    diag_acc = float((diag_pred == diag_label).mean())
    print(f"\n평가 (test 환자 {len(test_graphs)}명, {time.time() - t0:.1f}초 소요):")
    print(f"  주 과제  TR AUROC (합성): {auroc:.3f}")
    print(f"  보조 과제 진단 정확도 (합성): {diag_acc:.3f}")

    # adaptive hyperedge attention 가중치 (타입별 평균)
    alpha = out['alpha_e1'].numpy()
    etypes = out['edge_types'].numpy()
    emask = out['edge_mask'].numpy().astype(bool)
    print(f"\n학습된 adaptive hyperedge 가중치 (layer 1, 타입별 평균):")
    for t, name in EDGE_TYPE_NAMES.items():
        sel = (etypes == t) & emask
        if sel.sum():
            print(f"    {name:<18s}: {alpha[sel].mean():.3f}")

    print("\n※ 합성 데이터 결과는 파이프라인 동작 검증용이며 연구 수치가 아닙니다.")


# ============================================================================
# [2부] 실제 논문 실험 결과 출력
# ============================================================================
def show_thesis_results() -> None:
    banner("[2부] 실제 논문 실험 결과 (AI Hub #58: 환자 186명 / 세션 1,313개)")

    md_files = [
        ('C-block 2차: B4/B5/B6 그래프 모델 (5-fold × 5-seed)',
         ROOT / 'results/cblock2/cblock2_summary.md'),
        ('B6v2: 5번째 hyperedge (E_affective_pattern) 추가',
         ROOT / 'results/cblock2_b6v2/cblock2_summary.md'),
        ('Hyperedge ablation (타입별 기여 분석)',
         ROOT / 'results/hyperedge_ablation_v2/ablation_v2_summary.md'),
    ]
    for title, path in md_files:
        if path.exists():
            print(f"\n{'-' * 70}\n### {title}\n{'-' * 70}")
            print(path.read_text(encoding='utf-8').strip())

    youden = ROOT / 'results/youden_oof_multiseed/youden_oof_multiseed.json'
    if youden.exists():
        with open(youden, encoding='utf-8') as f:
            d = json.load(f)
        per_seed = d['per_seed']
        aurocs = [v['auroc'] for v in per_seed.values()]
        f1s = [v['youden_metrics']['f1_pos'] for v in per_seed.values()]
        recalls = [v['youden_metrics']['recall_pos'] for v in per_seed.values()]
        print(f"\n{'-' * 70}\n### Youden OOF 임계값 분석 (B6, {len(per_seed)} seeds, n=140)\n{'-' * 70}")
        print(f"  OOF AUROC          : {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
        print(f"  Youden 후 F1(pos)  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"  Youden 후 recall   : {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")

    print(f"\n{'-' * 70}")
    print("전체 파이프라인 재현에는 AI Hub #58/#539 데이터 승인이 필요합니다.")
    print("실행 절차: README.md 참조  |  코드: https://github.com/cmk77/k-ahfm")
    print(f"{'-' * 70}\n")


if __name__ == '__main__':
    banner("K-AHFM-Clinic — 심리상담 종단 기록 하이퍼그래프 멀티태스크 모델 (석사논문 데모)")
    print("""
  주 과제  : 치료반응(Treatment Response) 조기 예측 — 첫 3회기 → 전체 경과
  보조 과제: 4-class 진단 분류 (우울/불안/중독/정상)
  핵심 모델: B6 K-AHFM-Clinic — 세션 노드 하이퍼그래프
             + 4종 임상 hyperedge + adaptive attention weighting""")
    run_smoke_test()
    show_thesis_results()
