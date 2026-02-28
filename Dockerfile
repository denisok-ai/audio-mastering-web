# Magic Master — образ для production
# Сборка: docker build -t magic-master .
# Запуск: см. команды в DEPLOY-DOCKER.md

FROM python:3.11-slim-bookworm

# FFmpeg для MP3/FLAC (pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend
COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
# Статус плана для /progress.html
COPY PROGRESS.md ./PROGRESS.md

# Переменные по умолчанию (можно переопределить при run)
ENV MAGIC_MASTER_HOST=0.0.0.0
ENV MAGIC_MASTER_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Запуск из backend/, чтобы пакет app был в PYTHONPATH; фронтенд в /app/frontend
WORKDIR /app/backend
CMD ["python", "run_production.py"]
