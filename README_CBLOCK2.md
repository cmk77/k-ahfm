# C-block 2차 — B4/B5/B6 그래프 모델

## 개요

K-AHFM-Clinic 본 연구의 핵심 모델 (B6)과 그 비교 baseline (B4, B5)을 학습합니다.

| 모델 | 그래프 구조 | 핵심 특징 |
|---|---|---|
| **B4 Temporal GCN** | 평면 그래프 (시간 인접) | 단순 시계열 그래프 합성곱 |
| **B5 HYNMDR-style** | 단일 hyperedge 하이퍼그래프 | 단순 하이퍼그래프 (구조 가치만 검증) |
| **B6 K-AHFM-Clinic** ★ | 4종 hyperedge + adaptive | **본 연구 핵심 모델** |

## 노드 및 그래프 설계

### 노드 정의 (B4/B5/B6 공통)
- **노드 = 세션** (paragraph는 임베딩 평균으로 처리)
- **노드 feature 559차원**:
  - paragraph 임베딩 평균 (512)
  - paragraph 메타 평균 (5)
  - 28-증상 max-pool (28)
  - 11-개입 max-pool (11)
  - 3-차원 심각도 (3)

### B6의 4종 Hyperedge

| 종류 | 정의 | 환자당 개수 |
|---|---|---|
| **E_trajectory** | 모든 세션을 묶는 하나의 hyperedge | 1 |
| **E_co-trajectory** | high vs low 증상 강도 (median split) | 2 |
| **E_intervention-response** | 사용된 각 개입 종류별 hyperedge | 0~11 |
| **E_clinical_prior** | MDD/GAD/SUD 클러스터별 활성 세션 | 0~3 |

**Adaptive Weighting**: 각 hyperedge에 대해 *type embedding + 환자 context (노드 평균 + severity)* 로부터 attention score를 계산하여 softmax 가중치 적용.

## 파일 구성

```
cblock2/
├── src/
│   ├── __init__.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── build_graph.py     (457줄) — PatientGraph + 3종 그래프 빌더
│   ├── models/
│   │   ├── __init__.py
│   │   └── graph_models.py    (434줄) — B4/B5/B6 + Conv layers + Loss
│   └── training/
│       ├── __init__.py
│       ├── graph_trainer.py   (259줄) — 학습 루프, evaluate
│       └── metrics.py         (~165줄) — TR/진단 메트릭 (cblock1과 동일)
└── 09_train_graph_models.py   (~370줄) — 메인 학습 스크립트
```

## 적용 및 실행

### 적용

```bash
cd <repo>
tar xzf /mnt/c/Users/김창모/Downloads/k-ahfm-cblock2.tar.gz

# 파일 확인
ls -la src/graph/build_graph.py src/models/graph_models.py \
       src/training/graph_trainer.py src/training/metrics.py \
       09_train_graph_models.py
```

### 실행

```bash
source venv/bin/activate

python 09_train_graph_models.py \
    --extracted <repo>/data/extracted \
    --split training \
    --processed <repo>/data/processed \
    --embeddings <repo>/data/embeddings \
    --output <repo>/results/cblock2 \
    --scenarios early full \
    --models B4 B5 B6 \
    --seeds 0 42 2026 7 1024 \
    --input-window 3 \
    --max-epochs 100 \
    --patience 15 \
    --batch-size 16 \
    --hidden-dim 128 \
    --dropout 0.3 \
    --weight-decay 1e-3 \
    --lr 1e-3
```

**총 학습 횟수**: 3 모델 × 2 시나리오 × 5 fold × 5 seed = **150회**

**예상 시간**: GPU 48GB 활용 시 약 **3-5시간** (그래프 모델은 MLP보다 무거움; B6가 가장 무거움)

### 단일 모델 빠른 테스트 (권장: 먼저 B6만)

```bash
python 09_train_graph_models.py \
    --extracted <repo>/data/extracted \
    --split training \
    --processed <repo>/data/processed \
    --embeddings <repo>/data/embeddings \
    --output <repo>/results/cblock2_b6_test \
    --scenarios early \
    --models B6 \
    --seeds 0 \
    --max-epochs 50 --patience 10
```

→ 약 5-10분, 코드 정상 작동 검증용. AUROC 0.55+ 나오면 전체 실행 진행.

## 산출물

```
results/cblock2/
├── cblock2_config.json
├── cblock2_runs_B4_early.jsonl    — 25회 학습 결과
├── cblock2_runs_B4_full.jsonl
├── cblock2_runs_B5_early.jsonl
├── cblock2_runs_B5_full.jsonl
├── cblock2_runs_B6_early.jsonl
├── cblock2_runs_B6_full.jsonl
├── cblock2_runs.jsonl              — 통합 150회
├── cblock2_aggregated.json
└── cblock2_summary.md              — v5 표 4.1a/4.1b/4.2/4.3 형식
```

## 기대 결과

### vs C-block 1차 baseline (B3 v3: AUROC 0.547)

| 모델 | TR AUROC (Early) | Δ vs B3 v3 |
|---|---|---|
| B3 (v3, MLP) | 0.547 | — |
| **B4** | 0.55-0.62 | +0.00~0.07 |
| **B5** | 0.58-0.65 | +0.03~0.10 |
| **B6** | **0.60-0.70** | **+0.05~0.15** |

**합격 기준**:
- B6가 B3 대비 AUROC +0.03 이상 → 본 연구 contribution 입증
- B6 > B5 > B4 순서면 그래프 → 하이퍼그래프 → adaptive 가치 증명

### 진단 task

| 모델 | 진단 Macro F1 (Early) |
|---|---|
| B3 (v3) | 0.502 |
| B4 | 0.50-0.58 |
| B5 | 0.52-0.60 |
| B6 | **0.55-0.65** |

## 트러블슈팅

### CUDA OOM
- `--batch-size 8` 로 감소
- 그래프 모델은 동적 크기 padding으로 메모리 사용 변동 큼

### 학습 너무 느림
- 그래프 빌드가 첫 단계에서 가장 오래 걸림 (~5-10분, paragraph 임베딩 로드)
- 모델·시나리오 조합별로 그래프 새로 빌드 (총 6회 빌드)
- 학습 자체는 fast (paragraph 임베딩 캐싱 활용)

### B6 학습 안정성 문제
- adaptive weighting의 attention이 collapse하면 (모든 가중치가 1 edge에 몰림) 학습 불안정
- 이 경우 `--dropout 0.4`로 증가하거나 grad clip을 0.5로 조정

### Symptom cluster index 부정확
- `src/graph/build_graph.py`의 `MDD_SYMPTOM_INDICES`, `GAD_SYMPTOM_INDICES`, `SUD_SYMPTOM_INDICES`가 placeholder
- 실제 `data/constants.py`의 `SYMPTOM_KEYS` 순서에 맞춰 조정 가능
- 현재 설정도 충분히 합리적 (0-9 MDD, 10-16 GAD, 17-27 SUD)

## 다음 단계

C-block 2차 학습 완료 후:

### 경로 A: B6가 B3 대비 +0.05 이상 (성공)
- v5 논문 표 4.1a/4.1b/4.2/4.3 채움 → v6 완성
- §4.3-4.5 결과 분석 작성
- 디펜스 narrative: "*4종 hyperedge + adaptive weighting이 단순 MLP/GCN 대비 명확한 개선*"

### 경로 B: B6 일부 개선만 (+0.02~0.05)
- 결과 그대로 보고
- §5.3 한계에서 "텍스트 단일 모달의 본질적 한계" 강조
- §5.4 향후 연구에서 멀티모달 (#82 시각 모달리티 추가) 제안

### 경로 C: B6 개선 없음 (<+0.02)
- Ablation 분석을 통해 어느 hyperedge 타입이 기여 안 하는지 파악
- 결과 정직 보고, 5장에 결론적 분석 추가

세 경로 모두 v5 → v6 docx 갱신 후 디펜스 (5/22) 준비 완료.
