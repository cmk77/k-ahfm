# C-block 1차 v2 패치 — 시계열 보존 Feature

## 문제 진단

v1 결과의 핵심 문제는 **feature 추출에서 시계열 정보가 완전히 소실**된 것이었습니다.

```python
# v1 코드 (문제):
para_emb_patient = np.mean([sf['para_emb_mean'] for sf in session_features], axis=0)
```

이로 인해:
- TR AUROC: B3 = 0.550 (기대 0.62-0.72, MCC 0.047 ≈ random)
- 진단 macro F1: B3 = 0.424 < B1 = 0.503 (역설: feature 추가가 성능 악화)
- Full < Early (Full scenario가 더 나쁨, 시계열이 노이즈로 작용)

**근본 원인**: TR 라벨이 `S_late - S_early ≤ -1.0`로 정의되어 *시간 변화*가 핵심인데, 입력에서 시간 차원을 평균으로 지워버려 모델이 변화 신호를 학습할 수 없었습니다.

## v2 핵심 변경

**1. 시계열 블록 분리 집계** (`features.py`):

```python
def split_into_blocks(sessions):
    n = len(sessions)
    if n >= 6:    return sessions[:3], sessions[-3:]     # 라벨 정의와 일치
    if n in (4,5): half = n//2; return sessions[:half], sessions[-half:]
    if n == 3:    return sessions[:1], sessions[-1:]
    ...

def extract_patient_feature(...):
    first_block, last_block = split_into_blocks(sessions_sorted)
    first_feat = aggregate_block_features(first_block, ...)
    last_feat = aggregate_block_features(last_block, ...)
    all_feat = aggregate_block_features(sessions_sorted, ...)
    
    delta_emb = last_feat['para_emb'] - first_feat['para_emb']
    # ...
    
    b1 = concat([all_feat['para_emb'], delta_emb])  # 1024차원
```

**2. 새 차원** (`baselines.py`):
- B1: 512 → **1024** (mean_emb + delta_emb)
- B2: 517 → **1034** (+ mean_meta + delta_meta)
- B3: 576 → **1090** (+ mean_symptom_28 + delta_symptom_28)

**3. MLP 정규화 강화** (`baselines.py`):
- `nn.LayerNorm(input_dim)` 입력층 추가 (paragraph emb/meta/symptom scale 차이 흡수)
- `hidden_dim` 256 → **128** (overparameterization 완화)
- `dropout` 0.3 → **0.4** (정규화 강화)

## 적용 방법

```bash
cd <repo>

# v1 백업
cp src/utils/features.py backup/features_v1.py 2>/dev/null
cp src/models/baselines.py backup/baselines_v1.py 2>/dev/null

# v2 패치 적용
tar xzf /mnt/c/Users/김창모/Downloads/k-ahfm-cblock1-v2.tar.gz

# 적용 확인
grep -c "split_into_blocks" src/utils/features.py
# → 2 (정의 1번 + 호출 1번)

grep -c "nn.LayerNorm" src/models/baselines.py
# → 5 (입력 + backbone 2개 LayerNorm × 2 layers)

grep "B1_FEATURE_DIM" src/utils/features.py
# → "B1_FEATURE_DIM = 2 * EMBEDDING_DIM   # 1024"

# 이전 결과 보존 후 새로 실행
mv results/cblock1 results/cblock1_v1 2>/dev/null
```

## 재실행

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
    --hidden-dim 128 \
    --dropout 0.4 \
    --weight-decay 1e-3 \
    --lr 1e-3
```

**핵심 하이퍼파라미터 변경**:
- `--hidden-dim 128` (기본 256)
- `--dropout 0.4` (기본 0.3)
- `--weight-decay 1e-3` (기본 1e-4)

**예상 소요 시간**: 약 50-90분 (v1과 동일)

## 기대 결과 (v2)

| 모델 | TR AUROC (Early) | TR F1(pos) | 진단 macro F1 |
|---|---|---|---|
| B1 (v2) | 0.55-0.65 | 0.30-0.45 | 0.55-0.70 |
| B2 (v2) | 0.58-0.68 | 0.32-0.47 | 0.58-0.72 |
| B3 (v2) | **0.62-0.72** | **0.40-0.55** | **0.70-0.85** |

**중요 가설 검증**:
- B3 ≥ B2 ≥ B1 (feature 추가가 성능 *향상*으로 이어져야 함, v1과 반대)
- Full scenario ≥ Early scenario (full에서 진짜 delta 계산 가능)
- 진단 macro F1 0.70+ (28-증상 features가 정상 활용되면)

## 만약 v2도 미흡한 경우 (Plan B)

만약 v2 결과도 다음 기준에 못 미친다면:
- 진단 macro F1 < 0.65
- TR AUROC < 0.55

**추가 진단 옵션**:
1. **f_T 임베딩 재학습** (epoch 10-15로 늘려서 val_acc 0.50-0.55 목표)
2. **B0 추가** — paragraph 임베딩 무사용, 28-증상 features only (sparse baseline)
3. **Ablation** — 텍스트만 vs 28-증상만 vs 둘 다 비교

이 경우 결과 공유 후 추가 분석 진행하겠습니다.
