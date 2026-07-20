# K-AHFM-Clinic 데모 이미지 (CPU 전용)
#
# 빌드:  docker build -t k-ahfm .
# 실행:  docker run --rm k-ahfm
#
# AI Hub 원본 데이터는 라이선스상 이미지에 포함할 수 없으므로,
# 데모는 합성 데이터로 K-AHFM 하이퍼그래프 모델의 학습 파이프라인을
# 검증하고, 리포지토리에 저장된 실제 논문 실험 결과를 함께 출력한다.

FROM python:3.11-slim

WORKDIR /app

# CPU 빌드 PyTorch — 데모는 GPU 불필요
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy==1.26.4

# 소스 코드 + 실험 결과 (.dockerignore가 대용량 데이터·웨이트 제외)
COPY . .

CMD ["python", "demo.py"]
