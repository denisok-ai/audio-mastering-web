#!/bin/bash
# Запуск Magic Master: при необходимости установит пакеты (один раз запросит пароль) и запустит сервер.
# После запуска откройте в браузере: http://localhost:8000

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 1. Системные пакеты — если нет, устанавливаем (попросит пароль один раз)
MISSING=""
command -v ffmpeg &>/dev/null || MISSING="$MISSING ffmpeg"
ldconfig -p 2>/dev/null | grep -q "libatomic.so.1" || MISSING="$MISSING libatomic1"

if [ -n "$MISSING" ]; then
  echo "Устанавливаю пакеты:$MISSING (потребуется пароль)."
  sudo apt-get update -qq
  sudo apt-get install -y ffmpeg libatomic1
fi

# 2. Виртуальное окружение
cd "$ROOT/backend"
if [ ! -d "venv" ]; then
  echo "Создаю виртуальное окружение..."
  if ! python3 -m venv venv 2>/dev/null; then
    echo "Установите: sudo apt install -y python3.10-venv"
    exit 1
  fi
fi

# 3. Зависимости Python
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt

# 4. Запуск
echo ""
echo "Сервер запущен. Откройте в браузере: http://localhost:8000"
echo "Остановка: Ctrl+C"
echo ""
exec ./venv/bin/python run.py
