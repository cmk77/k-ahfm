# K-AHFM-Clinic — 2단계: 환경 구축 + 데이터 파이프라인

이 패키지는 K-AHFM-Clinic 석사 논문 연구의 2단계 작업물을 포함한다.
WSL2 Ubuntu + RTX 6000 Ada 환경에서 AI Hub #58 심리상담 데이터를 가공하여
이후 학습에 사용할 PyTorch Dataset까지 구축한다.

## 환경 요구사항

- WSL2 Ubuntu 22.04+ (또는 네이티브 Ubuntu 22.04+)
- NVIDIA Driver 570+ (RTX 6000 Ada 호환)
- CUDA 12.8 (Driver에 포함)
- Python 3.11
- 디스크 여유 공간: 약 20GB (가상환경 + 임베딩 캐시 + 처리 데이터)

## 디렉토리 구조

```
~/projects/k-ahfm-clinic/        ← PROJECT_DIR
├── 00_setup_env.sh              ← 1단계: 환경 구축
├── 01_extract_data.py           ← 2단계: ZIP 압축 해제
├── 02_verify_data.py            ← 3단계: 데이터 무결성 검증
├── 03_build_dataset.py          ← 4단계: A1-A5 파이프라인
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── constants.py         ← 28 증상요인 키 등 상수
│       ├── parser.py            (A1)
│       ├── normalize.py         (A2)
│       ├── prosodic.py          (A3)
│       ├── splits.py            (A4)
│       └── dataset.py           (A5)
├── data/                        ← 데이터 작업 공간 (자동 생성)
│   ├── extracted/               ← ZIP 해제 결과
│   ├── processed/               ← A1-A5 가공 결과
│   ├── cache/                   ← HuggingFace 모델 캐시
│   └── embeddings/              ← 사전계산 임베딩 (다음 단계)
├── venv/                        ← Python 가상환경
└── logs/
```

## 실행 순서

### Step 1: 패키지를 WSL2에 복사

WSL2 Ubuntu 쉘에서:

```bash
# 다운로드한 압축 파일을 풀어 ~/projects/k-ahfm-clinic 로 옮김
mkdir -p ~/projects
cd ~/projects
# (Windows 다운로드 폴더에서 풀어둔 폴더를 복사하거나)
mv /mnt/c/Users/김창모/Downloads/k-ahfm  ~/projects/k-ahfm-clinic
cd ~/projects/k-ahfm-clinic
chmod +x 00_setup_env.sh
```

### Step 2: 환경 구축

```bash
./00_setup_env.sh
```

이 스크립트는 자동으로 수행한다:
1. apt 패키지 설치 (Python 3.11, p7zip, locales 등)
2. 프로젝트 디렉토리 생성
3. nvidia-smi로 GPU 확인
4. Python 가상환경 생성 (venv/)
5. PyTorch 2.3.1 + CUDA 12.1 빌드 설치
6. requirements.txt 기반 패키지 설치
7. torch-geometric 설치

소요 시간: 5–10분.

완료 후 가상환경 활성화:

```bash
source ~/projects/k-ahfm-clinic/venv/bin/activate
```

### Step 3: 데이터 압축 해제

```bash
cd ~/projects/k-ahfm-clinic
python 01_extract_data.py \
    --source ~/datasets/aihub_58 \
    --dest ~/projects/k-ahfm-clinic/data/extracted
```

이 스크립트는:
- `~/datasets/aihub_58/`에서 Training/Validation × original/labeling 4-way 디렉토리 자동 감지
- 각 ZIP 파일을 압축 해제 (한글 파일명 CP949 인코딩 자동 처리)
- 진단군별 / 회기별로 구조화하여 저장
- 중복 실행 시 이미 해제된 파일은 건너뜀

미리 확인하려면 `--dry-run` 옵션 사용:

```bash
python 01_extract_data.py --source ~/datasets/aihub_58 --dest ~/projects/k-ahfm-clinic/data/extracted --dry-run
```

소요 시간: 2–5분 (텍스트만이라 가벼움).

### Step 4: 데이터 검증

```bash
python 02_verify_data.py --extracted ~/projects/k-ahfm-clinic/data/extracted
```

검증 항목:
- 진단군별 세션 수 (사용자 제공 정보 50/46/48과 비교)
- 환자별 회기 수 분포 (종단 데이터 검증)
- 28 증상요인 점수 분포 (0이 너무 많은가, 변동성이 있는가)
- 연속 회기 간 증상 변화 (종단 신호 강도)

### Step 5: A1-A5 데이터 파이프라인

```bash
python 03_build_dataset.py \
    --extracted ~/projects/k-ahfm-clinic/data/extracted \
    --output ~/projects/k-ahfm-clinic/data/processed \
    --n-folds 5 --seed 42
```

산출물:
- `data/processed/sessions.pkl` — Session 객체 리스트
- `data/processed/longitudinal_samples.pkl` — sliding window 종단 샘플
- `data/processed/fold_indices.json` — 환자 단위 5-fold 분할
- `data/processed/statistics.json` — 데이터 통계 요약

소요 시간: 3–10분 (JSON 파싱 위주).

## A1-A5 모듈 요약

| 모듈 | 역할 | 핵심 함수/클래스 |
|---|---|---|
| `src/data/constants.py` | 28 증상키, DSM-5 클러스터, 진단군 매핑 | `SYMPTOM_KEYS`, `DSM5_CLUSTERS`, `DIAGNOSIS_TO_INDEX` |
| `src/data/parser.py` (A1) | JSON → Session/Paragraph 객체 | `parse_session_json`, `parse_all_sessions` |
| `src/data/normalize.py` (A2) | 0-3 점수 → [0,1] 정규화, 세션 집계 | `session_symptoms_28`, `session_full_features` |
| `src/data/prosodic.py` (A3) | 음성 부재 환경의 prosodic proxy | `paragraph_prosodic_vector`, `compute_session_prosodic_stats` |
| `src/data/splits.py` (A4) | 환자 단위 5-fold + sliding window | `patient_disjoint_kfold`, `build_longitudinal_samples` |
| `src/data/dataset.py` (A5) | PyTorch Dataset | `K58SessionDataset`, `K58LongitudinalDataset`, `longitudinal_collate_fn` |

## 다음 단계 (3단계 예고)

A1-A5 완료 후 다음 작업으로 진행:

1. **B-block**: #539 텍스트 정서 인코더 사전학습 (Stage 0 Phase 1)
2. **임베딩 사전계산**: 모든 paragraph에 대해 KLUE-RoBERTa 임베딩을 한 번 계산하여 디스크 캐싱
3. **C-block**: B1-B5 baseline 학습
4. **D-block**: K-AHFM-Clinic 본 모델 학습

## 문제 해결

### Python 3.11 설치 실패

Ubuntu 22.04 기본 저장소에 python3.11이 없는 경우 deadsnakes PPA가 자동 추가됨.
실패 시 수동:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### CUDA 동작 실패

WSL2에서 GPU가 보이지 않는 경우:
1. Windows에서 최신 NVIDIA GPU 드라이버 (570+) 설치 확인
2. WSL2 커널 업데이트: `wsl --update` (PowerShell)
3. WSL2 재시작: `wsl --shutdown` → 다시 진입

### 한글 파일명 깨짐

`01_extract_data.py`는 CP437 → CP949 자동 변환을 시도. 그래도 깨지면:
- `LANG=ko_KR.UTF-8 python 01_extract_data.py ...`
- 또는 7z 사용: `7z x "TL_001. 우울증_0001. 1회기.zip" -mcp=949`

### torch-geometric 설치 실패

`pip install torch-geometric` 단독 실행으로 재시도. 캐시 정리:
```bash
pip cache purge
pip install torch-geometric==2.5.3 --no-cache-dir
```

## 라이선스 / 출처

- AI Hub #58 심리상담 데이터: AI Hub 정책 준수 (학술 연구용)
- KLUE-RoBERTa: Park et al. (2021) NeurIPS Datasets and Benchmarks
- PyTorch Geometric: Fey & Lenssen (2019)
