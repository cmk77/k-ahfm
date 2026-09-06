# K-AHFM-Clinic

심리상담(CBT) 종단 기록을 환자별 **세션 하이퍼그래프**로 구성하고, **adaptive hyperedge attention**으로
**치료반응(TR) 조기 예측 + 4-class 진단 분류**를 동시에 학습하는 멀티태스크 모델 (석사학위 논문 프로젝트).

- 데이터: AI Hub #58 심리상담 데이터 (환자 186명 / 세션 1,313개), AI Hub #539 멀티모달 감성 데이터 (f_T 사전학습)
- 평가: 환자 단위 5-fold × 5-seed 교차검증, early(첫 3회기) / full(전체 회기) 시나리오

## 빠른 데모 (데이터 불필요)

AI Hub 데이터 승인 없이, 설치 3줄로 바로 돌려볼 수 있습니다.

```bash
pip install torch "numpy<2" scikit-learn
python demo.py
```

출력은 두 부분입니다 — **합성 코호트로 B6 모델의 하이퍼그래프 구성·멀티태스크 학습을
end-to-end 실행하는 스모크 테스트**(수치 자체는 연구 결과가 아님)와, **저장소에 담긴
실제 논문 실험 결과(환자 186명 / 세션 1,313개, 5-fold × 5-seed)의 요약표**입니다.

## 1. Docker로 바로 실행 (권장)

설치 없이 데모를 바로 실행할 수 있습니다. 데모는 합성 데이터로 B6 모델(하이퍼그래프 구성 → 멀티태스크
학습 → adaptive attention 출력)을 스모크 테스트하고, 실제 논문 실험 결과 요약을 출력합니다.

```bash
docker pull ghcr.io/cmk77/k-ahfm:latest
docker run --rm ghcr.io/cmk77/k-ahfm:latest
```

> AI Hub 원본 데이터는 라이선스상 이미지에 포함되지 않습니다.

## 2. 전체 파이프라인 실행 (AI Hub 데이터 승인 필요)

환경: Ubuntu 22.04(WSL2 가능) + Python 3.11 + NVIDIA GPU(CUDA 12.x)

```bash
./00_setup_env.sh                      # 가상환경 + PyTorch 등 설치
source venv/bin/activate
```

| 단계 | 명령 | 내용 |
|---|---|---|
| 1 | `python 01_extract_data.py --source <aihub_58> --dest data/extracted` | 데이터 압축 해제 |
| 2 | `python 02_verify_data.py --extracted data/extracted` | 무결성 검증 |
| 3 | `python 03_build_dataset.py --extracted data/extracted --output data/processed` | 세션 파싱·전처리 |
| 4 | `python 04_generate_labels.py --extracted data/extracted --output data/processed` | TR 라벨 + 환자 단위 5-fold |
| 5 | `python 05_build_539_corpus.py --source <aihub_539> --output data/processed/aihub539_corpus.jsonl` | f_T 사전학습 corpus |
| 6 | `python 06_pretrain_text_emotion.py --corpus data/processed/aihub539_corpus.jsonl --output checkpoints/f_T` | f_T (KLUE/RoBERTa) 사전학습 |
| 7 | `python 07_precompute_paragraph_embeddings.py --extracted data/extracted --checkpoint checkpoints/f_T/best.pt --output data/embeddings` | 문단 임베딩 캐싱 |
| 8 | `python 08_train_baselines.py --extracted data/extracted --processed data/processed --embeddings data/embeddings --output results/cblock1` | B1–B3 MLP baseline |
| 9 | `python 09_train_graph_models.py --extracted data/extracted --processed data/processed --embeddings data/embeddings --output results/cblock2 --models B4 B5 B6` | B4–B6 그래프/하이퍼그래프 모델 |
| 10 | `python 12_hyperedge_ablation_v2.py ...` | hyperedge 타입별 ablation |
| 11 | `python 13c_youden_oof_multiseed.py ...` | OOF Youden 임계값 분석 |

각 스크립트는 `--help`로 전체 옵션을 확인할 수 있습니다.
학습 결과·집계 표는 `results/` 아래에 저장되어 있습니다.

## 데이터

이 저장소에는 **원데이터가 포함되어 있지 않습니다.** 공개 범위는 모델·실험 코드와
집계된 결과(`results/`)뿐입니다.

| 데이터 | 용도 | 접근 |
|---|---|---|
| AI Hub #58 심리상담 | 본 실험 (환자 186명 / 세션 1,313개) | AI Hub 이용 신청·승인 필요 |
| AI Hub #539 멀티모달 감성 | `f_T` 텍스트 감정 인코더 사전학습 | AI Hub 이용 신청·승인 필요 |

- 상담 원문은 민감정보이고 AI Hub 이용 정책상 재배포가 불가능합니다.
- 따라서 **`00_setup_env.sh` ~ `13c_*.py` 번호 파이프라인은 데이터 승인 이후에만 실행 가능**합니다.
  승인 없이 확인하려면 위의 [빠른 데모](#빠른-데모-데이터-불필요)를 쓰세요.
- 승인 후 절차는 [2. 전체 파이프라인 실행](#2-전체-파이프라인-실행-ai-hub-데이터-승인-필요)의 표를 따릅니다.

## 재검증 절차

논문 표에 실린 수치가 `results/`의 집계 JSON과 실제로 일치하는지 확인하는 스크립트들입니다.
모두 **인자 없이** 저장소 루트에서 실행하며, 논문 보고값을 코드 안에 상수로 박아 두고
집계 결과와 대조해 불일치를 출력합니다.

| 스크립트 | 역할 |
|---|---|
| `verify_tables.py` | 표 3a/3b(B1–B6 성능)를 `results/cblock1/`·`results/cblock2/`의 집계 JSON과 대조 |
| `verify_abl.py` | 표 6a/6b(hyperedge ablation) 대조 **1차판**. 조건 키를 `minus_E_co_trajectory`(언더스코어)로 가정하는데 현재 집계 JSON은 `minus_E_co-trajectory`(하이픈)라 `KeyError`로 중단됩니다 — `verify_abl2.py`로 대체됨 |
| `verify_abl2.py` | 같은 표 6a/6b 대조 **2차판**. 조건 키를 정규화(구두점·대소문자 제거)해 매칭하므로 표기 흔들림과 무관하게 동작 |
| `verify_alpha.py` | hyperedge 타입별 attention 가중치(α)를 `results/b6_attention_fold0_seed42_early/`의 환자별 npz에서 재계산해 논문값과 대조 |
| `sanity_check_v3.py` | 파이프라인 실행 **전** 점검 — 세션 객체에 `depression_severity`·`anxiety_severity`·`addiction_severity` 필드가 그 이름 그대로 있는지 확인 (이름이 다르면 특징이 0으로 채워져 조용히 망가짐). `--extracted` 경로 필요 |

```bash
python verify_tables.py    # 표 3a/3b
python verify_abl2.py      # 표 6a/6b  (verify_abl.py 는 키 표기 불일치로 실패 — 2차판을 쓸 것)
python verify_alpha.py     # attention α / beta_v

# 데이터 승인 후, 파이프라인 실행 전 점검
python sanity_check_v3.py --extracted <추출 경로>
```

### 임계값 분석의 누수 교정 (Youden J)

`13_*` 계열은 **같은 분석을 세 번 다시 한 기록**입니다. 앞 단계의 데이터 누수를
뒤 단계가 교정하므로, 보고에 쓸 수치는 마지막 것입니다.

| 단계 | 스크립트 | 코호트 구성 | 문제 / 개선 |
|---|---|---|---|
| 1 | `13_youden_threshold.py` | fold0 단일 run의 **train+val+test 예측을 합쳐** 140명 | train 101명이 모델의 학습 데이터 → **누수** |
| 2 | `13b_youden_oof.py` | 5개 fold **각각의 test 예측만** 모아 140명 (5×28, 중복 없음) | 진짜 out-of-fold — 누수 제거 |
| 3 | `13c_youden_oof_multiseed.py` | 위 OOF 구성을 **5개 시드**(0·42·2026·7·1024)에 반복 | 시드 분산까지 반영, 평균 ± 95% CI 보고 |

```bash
python 13b_youden_oof.py \
    --fold0 results/b6_attention_fold0_seed42_early \
    --oof-root results \
    --oof-pattern 'b6_oof_fold{F}_seed42_early' \
    --output results/youden_oof_seed42

python 13c_youden_oof_multiseed.py \
    --root results \
    --pattern 'b6_oof_fold{F}_seed{S}_early' \
    --fold0-seed42 results/b6_attention_fold0_seed42_early \
    --seeds 0 42 2026 7 1024 \
    --output results/youden_oof_multiseed
```

임계값 보정 자체는 평가자 지적("Recall_pos=0.114 — 35명 중 4명만 식별")에 대한 대응으로
시작했습니다. 0.5 고정 임계값 대신 Youden's J로 임계값을 정했을 때 recall이 어떻게
변하는지를 **누수 없는 OOF 코호트에서** 제시하는 것이 목적입니다.
`demo.py`가 출력하는 OOF 요약(AUROC 0.526 ± 0.027)이 3단계 결과입니다.

## 라이선스 / 출처

- AI Hub #58·#539 데이터: AI Hub 이용 정책 준수 (학술 연구 목적, 원본 재배포 불가)
- KLUE-RoBERTa: Park et al. (2021)
