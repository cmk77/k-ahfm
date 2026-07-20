# K-AHFM-Clinic

심리상담(CBT) 종단 기록을 환자별 **세션 하이퍼그래프**로 구성하고, **adaptive hyperedge attention**으로
**치료반응(TR) 조기 예측 + 4-class 진단 분류**를 동시에 학습하는 멀티태스크 모델 (석사학위 논문 프로젝트).

- 데이터: AI Hub #58 심리상담 데이터 (환자 186명 / 세션 1,313개), AI Hub #539 멀티모달 감성 데이터 (f_T 사전학습)
- 평가: 환자 단위 5-fold × 5-seed 교차검증, early(첫 3회기) / full(전체 회기) 시나리오

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

## 라이선스 / 출처

- AI Hub #58·#539 데이터: AI Hub 이용 정책 준수 (학술 연구 목적, 원본 재배포 불가)
- KLUE-RoBERTa: Park et al. (2021)
