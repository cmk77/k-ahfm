# C-block 1차 — B1/B2/B3 단순 Baseline 학습

## 개요

K-AHFM-Clinic 본 모델(B6)과의 비교 기준이 되는 세 가지 단순 baseline을 학습합니다.

| 모델 | 입력 차원 | 추가 features |
|---|---|---|
| **B1** | 512 | KLUE 평균 임베딩 (텍스트 only) |
| **B2** | 517 | + paragraph 메타 (cps, char_count, duration, speaker) |
| **B3** | 576 | + 28-증상 max + 20-위험 + 11-개입 (세션 단위 max-pool) |

**공통 구조**: MLP (input → 256 → 128 → output) + multi-task head (TR binary + 진단 4-class)

**Multi-task 손실**:
- TR (focal loss, α=0.75, γ=2.0)
- 진단 (weighted cross-entropy)
- λ_TR=1.0, λ_diag=0.3

## 패키지 구성

```
cblock1/
├── src/
│   ├── models/
│   │   ├── baselines.py        — B1/B2/B3 MLP + multi-task loss
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py          — 학습 루프, evaluate, early stopping
│   │   ├── metrics.py          — TR (AUROC/F1/MCC/Bal Acc) + 진단 (macro F1) + 진단군별 분해
│   │   └── __init__.py
│   ├── utils/
│   │   ├── features.py         — 환자 단위 feature 추출 (B1/B2/B3 동시)
│   │   └── __init__.py
│   └── __init__.py
└── 08_train_baselines.py       — 메인 학습 스크립트
```

## 적용

```bash
cd <repo>
tar xzf /mnt/c/Users/김창모/Downloads/k-ahfm-cblock1.tar.gz

# 구조 확인
ls -la src/models/baselines.py src/training/trainer.py src/utils/features.py 08_train_baselines.py
```

## 실행

```bash
source venv/bin/activate

python 08_train_baselines.py \
    --extracted <repo>/data/extracted \
    --split training \
    --processed <repo>/data/processed \
    --embeddings <repo>/data/embeddings \
    --output <repo>/results/cblock1 \
    --scenarios early full \
    --models B1 B2 B3 \
    --seeds 0 42 2026 7 1024 \
    --input-window 3 \
    --max-epochs 100 \
    --patience 15 \
    --batch-size 32 \
    --lr 1e-3
```

**총 학습 횟수**: 3 모델 × 2 시나리오 × 5 fold × 5 seed = **150회**.
**예상 시간**: GPU 활용 시 각 run ~20-40초 → 총 **약 50-100분**.

## 산출물

```
results/cblock1/
├── cblock1_config.json          — 실험 설정
├── cblock1_runs_early.jsonl     — early 시나리오 75회 학습 결과
├── cblock1_runs_full.jsonl      — full 시나리오 75회 학습 결과
├── cblock1_runs.jsonl           — 통합 150회 결과
├── cblock1_aggregated.json      — 25회 반복 평균 ± 95% CI
└── cblock1_summary.md           — 표 4.1a/4.1b 형식 셀
```

## 기대 결과 범위

**임상 NLP 문헌 기반 예측치** (현실적):

| 모델 | TR AUROC (early) | TR F1(pos) | 진단 macro F1 |
|---|---|---|---|
| B1 (텍스트 only) | 0.55-0.65 | 0.25-0.40 | 0.65-0.75 |
| B2 (+ meta) | 0.57-0.67 | 0.27-0.42 | 0.66-0.76 |
| B3 (+ 28-증상) | **0.62-0.72** | **0.35-0.50** | **0.75-0.85** |

핵심 가설:
- **B3 >> B1**: 28-증상 정량 점수가 가장 강한 시그널
- **Full >> Early**: 더 많은 회기 → 더 정확한 예측 (그러나 차이는 작을 수 있음)
- B4-B6 (그래프/하이퍼그래프 모델)이 B3를 추가로 개선하면 본 연구의 contribution 입증

## 학습 후 검증

```bash
# 결과 확인
cat results/cblock1/cblock1_summary.md

# 학습 횟수 검증 (150 줄이어야 함)
wc -l results/cblock1/cblock1_runs.jsonl

# 메트릭 분포 확인
python -c "
import json
with open('results/cblock1/cblock1_runs.jsonl') as f:
    runs = [json.loads(l) for l in f if l.strip()]
print(f'총 {len(runs)} 학습 완료')
for m in ['B1','B2','B3']:
    for s in ['early','full']:
        subset = [r for r in runs if r['model_name']==m and r['scenario']==s]
        aurocs = [r['test_metrics']['auroc'] for r in subset]
        import numpy as np
        print(f'{m}-{s}: n={len(subset)}, AUROC mean={np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}')
"
```

## 다음 단계

C-block 1차 결과 보고 후 평가:
- B1-B3 결과가 기대 범위 내 → C-block 2차 (B4 Temporal GCN, B5 HYNMDR, B6 K-AHFM-Clinic) 진행
- B3가 너무 낮으면 (TR AUROC < 0.55) 데이터 분포 재검토 또는 학습 안정성 점검

## 문제 해결

### CUDA out of memory
batch_size를 16으로 줄이세요.

### 학습이 너무 느림
- `--num-workers 0` (multiprocessing overhead 제거 시도)
- 환자 feature 추출이 첫 단계에서 가장 시간 소요 (~5-10분), 그 후 학습은 빠름

### 결과가 너무 변동성 클 경우
- patience를 20으로 증가
- lr을 5e-4로 감소
