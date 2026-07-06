# B-block — #539 텍스트 정서 인코더 사전학습 + #58 임베딩 사전계산

## 패키지 구성

```
bblock/
├── src/
│   ├── data/
│   │   ├── aihub539.py              — #539 JSON 파서 (Utterance 추출)
│   │   └── text_emotion_dataset.py  — PyTorch Dataset (KLUE 토크나이저 인코딩)
│   └── models/
│       ├── __init__.py
│       └── text_encoder.py          — KoreanTextEmotionEncoder (KLUE + multi-task head)
├── 05_build_539_corpus.py           — Step 1: #539 발화 corpus 추출
├── 06_pretrain_text_emotion.py      — Step 2: f_T 사전학습 (KLUE/RoBERTa fine-tune)
└── 07_precompute_paragraph_embeddings.py  — Step 3: #58 paragraph 임베딩 캐싱
```

## 적용 방법

WSL2에서:

```bash
cd <repo>

# 1. 백업
mkdir -p backup
[ -f src/data/aihub539.py ] && cp src/data/aihub539.py backup/

# 2. 패치 적용
tar xzf /mnt/c/Users/김창모/Downloads/k-ahfm-bblock.tar.gz

# 3. 적용 확인
ls -la src/data/aihub539.py src/data/text_emotion_dataset.py \
       src/models/text_encoder.py 05_build_539_corpus.py \
       06_pretrain_text_emotion.py 07_precompute_paragraph_embeddings.py
```

## Step 1 — #539 발화 corpus 추출 (10–20분)

```bash
source venv/bin/activate

python 05_build_539_corpus.py \
    --source <외부 데이터 경로>/aihub_539_multimodal \
    --output <repo>/data/processed/aihub539_corpus.jsonl
```

기대 출력:

```
[aihub539] 발견된 clip JSON: ~6,000개
[aihub539] 총 ~85,000개 unique 발화 추출

  [주 라벨: emotion.multimodal]
    라벨 있음: ~80,000개 (~94%)
      happy     : ...
      sad       : ...
      angry     : ...
      ...
      neutral   : ...

  [발화 길이 (글자 수)]
    mean=~25, median=~20, max=~200

✓ Corpus 추출 완료: ~85,000개 발화
```

이 단계에서 `data/processed/aihub539_corpus.jsonl` (~30-50MB) 생성. 다음 단계에서 재사용.

## Step 2 — f_T 사전학습 (3–6 GPU시간)

```bash
python 06_pretrain_text_emotion.py \
    --corpus <repo>/data/processed/aihub539_corpus.jsonl \
    --output <repo>/checkpoints/f_T \
    --batch-size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --fp16 \
    --num-workers 4
```

**옵션 설명**:
- `--batch-size 64`: RTX 6000 Ada 48GB VRAM 활용 (KLUE-base + seq 128)
- `--epochs 5`: 일반적으로 3-5 epoch에 수렴
- `--lr 2e-5`: KLUE fine-tune 표준 learning rate
- `--fp16`: Mixed precision (속도 1.5-2배, VRAM 절약)

**기대 결과** (5 epochs 학습 후):

```
[epoch 1] train loss 1.85  val loss 1.62  acc 0.51  macro F1 0.42  V-r 0.45  A-r 0.42
[epoch 2] train loss 1.45  val loss 1.32  acc 0.58  macro F1 0.51  V-r 0.55  A-r 0.51
[epoch 3] train loss 1.20  val loss 1.18  acc 0.63  macro F1 0.57  V-r 0.62  A-r 0.58
[epoch 4] train loss 1.05  val loss 1.12  acc 0.66  macro F1 0.60  V-r 0.65  A-r 0.61
[epoch 5] train loss 0.94  val loss 1.10  acc 0.67  macro F1 0.62  V-r 0.67  A-r 0.62

✓ Best checkpoint: <repo>/checkpoints/f_T/best.pt
```

산출물 (`checkpoints/f_T/`):
- `best.pt` — best val accuracy 체크포인트
- `final.pt` — 마지막 epoch 체크포인트
- `tokenizer/` — KLUE 토크나이저 사본
- `training_log.txt`
- `metrics.json`

## Step 3 — #58 paragraph 임베딩 사전계산 (30–60분)

```bash
python 07_precompute_paragraph_embeddings.py \
    --sessions <repo>/data/processed/sessions.pkl \
    --checkpoint <repo>/checkpoints/f_T/best.pt \
    --output <repo>/data/embeddings \
    --batch-size 64 \
    --fp16 \
    --num-workers 4
```

만약 `sessions.pkl`이 없으면 `--extracted <repo>/data/extracted` 옵션을 사용.

**기대 결과**:
```
처리한 세션:     1,313개
총 임베딩 수:    ~80,000–100,000 paragraph
소요 시간:       30–60분
디스크 사용량:   ~200–300 MB  (512-dim float32)
```

산출물 (`data/embeddings/`):
- `D002_s01_para_emb.npy` (각 (N, 512) shape)
- `D002_s02_para_emb.npy`
- ... (세션별 약 1,300개 파일)

## 검증

학습 후 임베딩이 정상인지 빠르게 확인:

```bash
python -c "
import numpy as np
import glob

files = sorted(glob.glob('<repo>/data/embeddings/*_para_emb.npy'))
print(f'임베딩 파일: {len(files)}개')

if files:
    emb = np.load(files[0])
    print(f'첫 파일 {files[0].split(chr(47))[-1]}: shape={emb.shape}, dtype={emb.dtype}')
    print(f'  값 통계: mean={emb.mean():.4f}, std={emb.std():.4f}')
    print(f'  norm: mean={np.linalg.norm(emb, axis=1).mean():.4f}')

total_paragraphs = sum(np.load(f).shape[0] for f in files[:100])
print(f'\n첫 100세션 paragraph 합: {total_paragraphs:,}')
"
```

각 임베딩 벡터의 norm이 5-15 범위에 있고 std가 0.1-1.0 사이면 정상입니다.

## 다음 단계 — C-block

B-block 완료 후 C-block (B1-B5 baseline + B6 K-AHFM-Clinic 본 모델 학습)으로 진행합니다.
사전계산된 임베딩 덕분에 baseline·본 모델·ablation 학습이 paragraph 단위 forward 없이
*저장된 임베딩만 읽어서* 진행되므로 학습 시간이 수십 배 단축됩니다.

## 문제 해결

### Out of memory (OOM)
- `--batch-size 32` 또는 `--batch-size 16`으로 줄임
- `--max-length 96`로 줄임 (대부분 발화가 짧음)

### KLUE 다운로드 실패
HuggingFace 캐시를 명시적으로 설정:
```bash
export HF_HOME=<repo>/data/cache
export TRANSFORMERS_CACHE=<repo>/data/cache
```

### 학습이 너무 느림
- 한국 네트워크에서 KLUE 다운로드(약 500MB)는 처음 1번만 발생
- 학습 중 GPU 사용률 확인: `watch -n 2 nvidia-smi`
- 80% 이상이어야 정상
