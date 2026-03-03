#!/bin/bash
# Упаковка проекта для выгрузки на сервер (без venv, кэша и тестовых файлов).
# Запуск из корня проекта: ./pack_for_deploy.sh
# Создаёт: audio-mastering-web-deploy-YYYYMMDD.tar.gz в текущей папке.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DATE=$(date +%Y%m%d)
ARCHIVE="audio-mastering-web-deploy-${DATE}.tar.gz"

echo "Упаковка Magic Master для развёртывания..."
tar --exclude='backend/venv' \
    --exclude='backend/__pycache__' \
    --exclude='backend/app/__pycache__' \
    --exclude='*.pyc' \
    --exclude='backend/test_*.wav' \
    --exclude='backend/test_signal.wav' \
    --exclude='test_output' \
    --exclude='.git' \
    -czvf "$ARCHIVE" \
    backend \
    frontend \
    deploy \
    start.sh \
    pack_for_deploy.sh \
    README.md \
    DEPLOY.md \
    PROGRESS.md

echo ""
echo "Готово: $ROOT/$ARCHIVE"
echo "Скопируйте архив на сервер и следуйте инструкции в DEPLOY.md"
