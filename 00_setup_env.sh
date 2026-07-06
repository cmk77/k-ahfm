#!/usr/bin/env bash
# ============================================================================
# K-AHFM-Clinic 환경 구축 스크립트
# 대상 환경: WSL2 Ubuntu, RTX 6000 Ada (48GB), CUDA 12.8, Driver 570+
# ============================================================================

set -euo pipefail

# ----- 색상 출력 -----
GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ----- 경로 설정 -----
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/k-ahfm-clinic}"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_VERSION="3.11"

log "프로젝트 디렉토리: $PROJECT_DIR"
log "Python 버전: $PYTHON_VERSION"

# ----- 1. 시스템 패키지 설치 -----
log "[1/7] 시스템 패키지 업데이트 및 필수 도구 설치..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential git curl wget software-properties-common \
    p7zip-full unzip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    locales tzdata

# Python 3.11 설치 (Ubuntu 기본 저장소 또는 deadsnakes ppa)
if ! command -v python3.11 &> /dev/null; then
    log "Python 3.11 설치 중 (deadsnakes ppa)..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
fi

# 한글 locale (zip 파일명 처리용)
sudo locale-gen ko_KR.UTF-8 en_US.UTF-8
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8

# ----- 2. 프로젝트 디렉토리 구조 생성 -----
log "[2/7] 프로젝트 디렉토리 구조 생성..."
mkdir -p "$PROJECT_DIR"/{data/{extracted,processed,cache,embeddings},src/data,src/models,src/training,src/utils,scripts,configs,notebooks,outputs,logs}

# ----- 3. CUDA / GPU 확인 -----
log "[3/7] GPU/CUDA 동작 확인..."
if ! command -v nvidia-smi &> /dev/null; then
    err "nvidia-smi 명령을 찾을 수 없음. WSL2에서 NVIDIA 드라이버를 설치하세요."
    exit 1
fi
nvidia-smi | head -20

# ----- 4. Python 가상환경 생성 -----
log "[4/7] Python 가상환경 생성: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    warn "기존 venv 존재 — 그대로 사용합니다. 새로 만들려면 먼저 'rm -rf $VENV_DIR' 실행."
else
    python3.11 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ----- 5. PyTorch + CUDA 12.1 빌드 설치 -----
# CUDA 12.8 드라이버에서 cu121 빌드 호환 작동 (PyTorch 공식 권장)
log "[5/7] PyTorch 2.3.1 + CUDA 12.1 빌드 설치 (시간 소요: 약 3-5분)..."
pip install --quiet \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 동작 검증
log "PyTorch CUDA 검증:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  cuDNN version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name}')
    print(f'  VRAM: {p.total_memory / 1e9:.2f} GB')
    print(f'  Compute capability: {p.major}.{p.minor}')
"

# ----- 6. 나머지 패키지 설치 -----
log "[6/7] 나머지 Python 패키지 설치..."
REQ_FILE="$PROJECT_DIR/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    err "requirements.txt를 찾을 수 없음: $REQ_FILE"
    err "이 스크립트와 함께 제공된 requirements.txt를 위 경로에 두고 다시 실행하세요."
    exit 1
fi
pip install --quiet -r "$REQ_FILE"

# torch-geometric은 torch 설치 이후 설치 (의존성 때문)
log "torch-geometric 설치..."
pip install --quiet torch-geometric==2.5.3

# ----- 7. 설치 결과 요약 -----
log "[7/7] 설치된 패키지 요약:"
pip list 2>/dev/null | grep -iE "torch|transformers|tokenizers|numpy|pandas|sklearn|pyg|geometric" | head -20

cat <<EOF

${GREEN}═══════════════════════════════════════════════════════════════${NC}
${GREEN}  환경 구축 완료${NC}
${GREEN}═══════════════════════════════════════════════════════════════${NC}

다음 명령으로 가상환경 활성화:
  ${BLUE}source $VENV_DIR/bin/activate${NC}

다음 단계 (압축 해제):
  ${BLUE}cd $PROJECT_DIR${NC}
  ${BLUE}python 01_extract_data.py \\
      --source ~/datasets/aihub_58 \\
      --dest ~/projects/k-ahfm-clinic/data/extracted${NC}

EOF
