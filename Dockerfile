# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 캐시 효율
COPY requirements.txt /app/requirements.txt

# pip 업그레이드 + 의존성 설치
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# 코드 복사
COPY . /app

# 비루트 사용자 생성 (보안 권장)
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
