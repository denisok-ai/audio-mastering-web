#!/bin/bash
# @file install_and_run.sh
# @description Установка зависимостей и запуск Magic Master локально
# @created 2026-02-26

set -e
cd "$(dirname "$0")/backend"

echo "=== 1. Проверка системных зависимостей ==="
MISSING=""
if ! command -v ffmpeg &>/dev/null; then
    MISSING="${MISSING} ffmpeg"
fi
if ! ldconfig -p 2>/dev/null | grep -q "libatomic.so.1"; then
    MISSING="${MISSING} libatomic1"
fi
if [ -n "$MISSING" ]; then
    echo "Не найдены пакеты:$MISSING"
    echo "Установите их одной командой (потребуется пароль):"
    echo "  sudo apt update && sudo apt install -y ffmpeg libatomic1 python3.10-venv"
    echo "После установки снова запустите этот скрипт."
    exit 1
fi
echo "FFmpeg: $(ffmpeg -version | head -1)"
echo "libatomic: OK"

echo ""
echo "=== 2. Виртуальное окружение ==="
if [ ! -d "venv" ]; then
    if ! python3 -m venv venv 2>/dev/null; then
        echo "Не удалось создать venv. Установите пакет:"
        echo "  sudo apt install -y python3.10-venv"
        echo "Затем снова запустите этот скрипт."
        exit 1
    fi
    echo "Создано: backend/venv"
else
    echo "Используется существующее: backend/venv"
fi

echo ""
echo "=== 3. Установка зависимостей Python ==="
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
echo "Готово."

echo ""
echo "=== 4. Запуск сервера ==="
echo "Откройте в браузере: http://localhost:8000"
echo "Остановка: Ctrl+C"
echo ""
exec ./venv/bin/python run.py
