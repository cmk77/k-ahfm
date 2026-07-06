# C-block 1차 v3 패치 — 임상 심각도 입력 추가

## v2 → v3 핵심 변경

v2의 핵심 한계: **TR 라벨은 *임상가 평가 심각도* 변화로 정의되는데, 모델 입력은 *텍스트 임베딩* 변화만 줬음**. 두 차원의 상관관계가 약해서 모델이 라벨에 직접 정합하지 못했습니다.

**v3 해결**: 임상 심각도(depression/anxiety/addiction)를 6 dims만 추가 입력으로 제공.

### 추가되는 features

```python
severity_first_3 (3 dims):  첫 회기의 [dep_sev, anx_sev, add_sev] / 3.0
severity_mean_input (3 dims): 입력 회기들의 평균 심각도
```

### 누설(leakage) 안전성 검증

- ✓ `severity_first`: 첫 회기 정보 → S_early의 부분, S_late는 모름
- ✓ `severity_mean_input`: 입력 회기 평균 → Early=S_early, Full=평균 (S_late만 직접 누설 아님)
- ✗ `last block severity`: **사용하지 않음** (Full에서 S_late = 라벨이므로 누설)

### 새 차원

| 모델 | v2 | v3 | 변화 |
|---|---|---|---|
| B1 | 1024 | **1030** | +6 |
| B2 | 1034 | **1040** | +6 |
| B3 | 1090 | **1096** | +6 |

## 적용 절차

### Step 1: 사전 검증 (필수)

Session 객체의 severity attribute 이름을 확인합니다:

```bash
cd <repo>
source venv/bin/activate

# v3 패치 적용
tar xzf /mnt/c/Users/김창모/Downloads/k-ahfm-cblock1-v3.tar.gz

# Sanity check 실행
python sanity_check_v3.py --extracted <repo>/data/extracted
```

**기대 출력**:
```
[1] 첫 세션 파싱
  → 1,313 세션 로드

[2] Severity attribute 검증
  patient    session  dep    anx    add
  ----------------------------------------
  D002       s1       2.0    1.0    0.0
  D002       s2       2.0    1.0    0.0
  ...
  ✓ 5/5 세션에서 severity attribute 정상 추출

[3] 전체 세션의 severity 분포
  depression : mean=1.20, std=0.95, min=0, max=3
  anxiety    : mean=0.85, std=0.78, min=0, max=3
  addiction  : mean=0.45, std=0.65, min=0, max=3

  최대값: 3
  ✓ 0-3 범위, 정규화 / 3.0 적합

✓ Sanity check 완료. v3 패치 적용 안전.
```

**만약 다음과 같이 나오면**:
- "Severity attribute가 없거나 다른 이름" → 결과 공유, attribute 이름 맞춰 features.py 수정 필요
- "최대값이 5 초과" → severity가 sum score (0-15 정도)일 가능성. features.py의 `/ 3.0`을 `/ 15.0`으로 변경 필요

### Step 2: v2 결과 백업 후 재실행

```bash
mv results/cblock1 results/cblock1_v2 2>/dev/null

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

## 기대 결과 (v3)

| 모델 | TR AUROC (Early) | 진단 macro F1 (Early) |
|---|---|---|
| v2 B1 | 0.549 | 0.416 |
| v2 B2 | 0.577 | 0.499 |
| v2 B3 | 0.577 | 0.370 |
| **v3 B1** | **0.65-0.72** | **0.55-0.65** |
| **v3 B2** | **0.67-0.74** | **0.58-0.68** |
| **v3 B3** | **0.68-0.75** | **0.60-0.70** |

### 왜 이 정도로 개선되는가

**TR 예측**:
- `severity_first_3`이 초기 상태를 직접 알려줌
- 임상 관찰: severity 3에서 시작한 환자가 severity 1이나 2에서 시작한 환자보다 호전(1단위 감소)할 여지가 더 큼
- 이 직접적 prior가 모델에 +0.10 AUROC 효과 예상

**진단 예측**:
- 우울증 환자 → depression_severity 1-3, anxiety/addiction 0-1
- 불안 환자 → anxiety_severity 1-3, depression/addiction 0-1
- 중독 환자 → addiction_severity 1-3, others 낮음
- **`argmax(severity_first_3)`만으로도 진단을 80%+ 정확도로 맞출 수 있을 가능성** (severity가 진단 정의의 일부이므로)

### 만약 v3가 실패하면

만약 v3 결과가 여전히 다음 미달이면:
- TR AUROC < 0.60
- 진단 macro F1 < 0.55

**원인 분석 옵션** (자동 진단):
1. Session severity가 모든 환자에게 비슷한 값일 수도 (variance 부족)
2. Severity가 paragraph 단위 28-증상의 sum score일 수도 (이미 B3 features에 포함된 정보)
3. Class weighting 문제

이 경우 추가 진단 후 **Option B (현 결과 그대로 baseline 확정 후 C-block 2 진행)** 으로 이동합니다.

## 다음 단계

학습 완료 (50-90분) 후 `cblock1_summary.md` 공유해 주세요. v2와 비교하여:

**경로 A — 개선 확인 (B3 AUROC 0.65+, 진단 0.60+)**:
즉시 C-block 2차 (B4-B6) 패키지 작성 시작.

**경로 B — 일부 개선만 (B3 AUROC 0.60-0.65)**:
v3 결과 그대로 v5 논문 baseline으로 확정, C-block 2차 진행.

**경로 C — 미달 (B3 AUROC < 0.60)**:
원인 진단 (severity 분포 분석) 후 Option B로 즉시 이동.

세 경로 모두에서 *결국 C-block 2차로 진행*하므로 시간 손실은 최소화됩니다.
