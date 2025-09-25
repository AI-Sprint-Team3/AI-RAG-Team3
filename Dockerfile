# CUDA 기반 이미지 사용 → GPU 환경도 포함
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    build-essential \
    git curl \
    poppler-utils libgl1 \
    && rm -rf /var/lib/apt/lists/*

# requirements 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# 소스 복사
COPY . /app

# 비루트 사용자 설정
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
