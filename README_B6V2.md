# C-block 2차 (B6v2) — 5번째 hyperedge + emotion features

## v6 → v7 진화 핵심

원래 설계 (paper §3.4)를 *완성도 높여* 구현:

| 항목 | B6 (v1, 현재 v6 docx) | **B6v2 (v7 진행)** |
|---|---|---|
| 노드 feature | 559-dim | **569-dim** (+ 8 emotion logits + 2 V/A) |
| Hyperedge 타입 수 | 4 | **5** |
| 5번째 type | — | **E_affective_pattern** (4 hyperedge) |
| 이론 근거 | DSM-5 clinical prior | + Russell circumplex + Plutchik wheel |
| f_T 활용 | 임베딩만 (implicit) | 임베딩 + 분류 헤드 (explicit emotion) |

## E_affective_pattern — 4 hyperedge 정의

| Hyperedge | 조건 | 임상 의미 |
|---|---|---|
| **E_low_valence** | valence ≤ median | 부정 정서 (우울 신호) |
| **E_high_arousal** | arousal > median | 흥분/긴장 (불안 신호) |
| **E_distress** | low_val + high_arousal | 만성 디스트레스 |
| **E_emotional_volatility** | entropy > median | 감정 변동성 (중독 신호) |

## 파일 구성

```
cblock2_b6v2/
├── scripts/
│   └── extract_emotion_features.py  (260줄) — f_T 분류 헤드로 emotion + V/A 추출
├── src/
│   ├── graph/
│   │   ├── __init__.py
│   │   └── build_graph.py           (420줄) — B6v2 5-hyperedge 빌더
│   └── models/
│       ├── __init__.py
│       └── graph_models.py          (~250줄) — B6/B6v2 모델 (input_dim 자동)
└── 09_train_graph_models.py         (~320줄) — B6v2 학습 스크립트
```

## 적용 및 실행 — 2단계

### Step 1: emotion features 추출 (한 번만, ~5분)

```bash
cd <repo>
source venv/bin/activate

# 패키지 적용
tar xzf k-ahfm-cblock2-b6v2.tar.gz

# (선택) 먼저 f_T 체크포인트의 헤드 키 확인:
python scripts/extract_emotion_features.py \
    --checkpoint checkpoints/f_T/best.pt \
    --embeddings data/embeddings \
    --output data/emotion_features.json \
    --list-keys

# emotion features 추출 (자동 탐색)
python scripts/extract_emotion_features.py \
    --checkpoint checkpoints/f_T/best.pt \
    --embeddings data/embeddings \
    --output data/emotion_features.json
```

**예상 출력**:
```
체크포인트 로드: checkpoints/f_T/best.pt
state_dict 키 수: N
사용할 헤드 키:
  emo_w = classifier_emo.weight, shape=(8, 512)
  emo_b = classifier_emo.bias, shape=(8,)
  av_w = classifier_av.weight, shape=(2, 512)
  av_b = classifier_av.bias, shape=(2,)
임베딩 파일 1313개 처리 시작...
  [200/1313] processed
  ...
V/A 원본 범위:
  Valence: [-X.XXX, X.XXX]
  Arousal: [-X.XXX, X.XXX]
정규화 후 세션 V/A 분포:
  Valence_mean: mean=0.XXX, std=0.XXX
  Arousal_mean: mean=0.XXX, std=0.XXX
✓ 완료. 세션 1313개 emotion features 추출.
```

### Step 2: B6v2 학습 (~10-15분)

```bash
python 09_train_graph_models.py \
    --extracted <repo>/data/extracted \
    --split training \
    --processed <repo>/data/processed \
    --embeddings <repo>/data/embeddings \
    --emotion-features <repo>/data/emotion_features.json \
    --output <repo>/results/cblock2_b6v2 \
    --scenarios early full \
    --models B6v2 \
    --seeds 0 42 2026 7 1024 \
    --max-epochs 100 --patience 15 \
    --batch-size 16 --hidden-dim 128 \
    --dropout 0.3 --weight-decay 1e-3 --lr 1e-3
```

**총 학습 횟수**: 1 모델 × 2 시나리오 × 5 fold × 5 seed = **50회**, 약 10-15분

### Step 3 (선택): B6 vs B6v2 비교 학습

```bash
python 09_train_graph_models.py \
    --extracted ... --processed ... --embeddings ... \
    --emotion-features <repo>/data/emotion_features.json \
    --output <repo>/results/cblock2_b6_vs_b6v2 \
    --scenarios early full \
    --models B6 B6v2 \
    --seeds 0 42 2026 7 1024
```

## 기대 결과 (B6 → B6v2)

| 메트릭 | B6 (v6 측정치) | B6v2 (기대) | 입증 시 |
|---|---|---|---|
| TR AUROC (Early) | 0.599 | **0.60-0.66** | +0.01~0.06 |
| TR AUROC (Full) | 0.568 | **0.57-0.62** | +0.00~0.05 |
| 진단 macro F1 (Early) | 0.660 | **0.68-0.75** | +0.02~0.09 |
| 진단 macro F1 (Full) | 0.758 | **0.76-0.80** | +0.00~0.04 |
| 진단군별 우울 AUROC | 0.589 | **0.60-0.66** | +0.01~0.07 |
| 진단군별 불안 AUROC | 0.646 | **0.66-0.72** | +0.01~0.07 |
| 진단군별 중독 AUROC | 0.700 | **0.70-0.76** | +0.00~0.06 |

**최소 합격선**: B6v2가 B6 대비 진단 F1 +0.03 또는 TR AUROC +0.02 이상이면 5번째 hyperedge contribution 입증.

## 산출물

```
results/cblock2_b6v2/
├── cblock2_config.json
├── cblock2_runs_B6v2_early.jsonl  — 25회 학습 결과 (early)
├── cblock2_runs_B6v2_full.jsonl   — 25회 학습 결과 (full)
├── cblock2_runs.jsonl              — 통합 50회
├── cblock2_aggregated.json
└── cblock2_summary.md              — 표 형식 결과
```

## 트러블슈팅

### `extract_emotion_features.py`에서 헤드 키 미발견

```bash
# 키 목록 출력하여 확인
python scripts/extract_emotion_features.py \
    --checkpoint checkpoints/f_T/best.pt \
    --embeddings data/embeddings \
    --output data/emotion_features.json \
    --list-keys

# 출력에서 실제 키 이름 확인 후 명시:
python scripts/extract_emotion_features.py \
    --checkpoint checkpoints/f_T/best.pt \
    --embeddings data/embeddings \
    --output data/emotion_features.json \
    --emo-weight-key model.head_emotion.weight \
    --emo-bias-key model.head_emotion.bias \
    --av-weight-key model.head_av.weight \
    --av-bias-key model.head_av.bias
```

### 임베딩 차원이 512가 아님

`extract_emotion_features.py`는 자동으로 512 또는 768 차원 모두 지원. 그 외 차원이면 헤드 차원 확인 후 매칭.

### B6v2 학습이 unstable (validation auroc 진동)

- `--dropout 0.4` 로 증가
- `--lr 5e-4` 로 감소
- `--patience 20` 로 증가하여 학습 더 진행

## 다음 단계 — v6 → v7 docx 업데이트

B6v2 결과 받은 후:

### 시나리오 A (가능성 60%): B6v2가 B6 대비 +0.03 이상 개선
- v6 → **v7 docx 업데이트**:
  - 표 4.1a/4.1b/4.3에 B6v2 행 추가 (B4-B6 + B6v2)
  - 표 4.2에 B6v2 진단군별 TR 분해 추가
  - §3.4.5 신설 (E_affective_pattern 설명)
  - §4.3 narrative 강화 ("5종 hyperedge가 4종 대비 +X% 개선")
  - §5.1 contribution 5번째 항목 강화 (정량 입증)
  - §5.3 한계 한 항목 제거 (B5/B6 동등성 약화)

### 시나리오 B (가능성 25%): B6v2가 B6와 통계적 동등 (+0.00~0.03)
- v7 docx: 결과 그대로 보고
- §4.5 ablation 항목 ("E_affective_pattern은 통계적 동등 수준 기여")
- §5.3 한계 유지 ("코호트 규모로 인한 5번째 hyperedge 가치 검증 한계")

### 시나리오 C (가능성 15%): B6v2가 B6보다 약화
- 디버그: E_affective_pattern 정의 점검 (threshold 조정)
- 또는 V/A 정규화 점검
- 또는 노드 차원 559 → 569 자체의 dropout 조정
- 30-60분 재실험

5/18 디펜스 데드라인 기준 5일 남았으므로 B6v2 결과 확인 후 *경로 결정 즉시* v7 작성 진행.
