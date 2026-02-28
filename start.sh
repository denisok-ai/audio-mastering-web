#!/bin/bash
# Запуск Magic Master: при необходимости установит пакеты (один раз запросит пароль) и запустит сервер.
# После запуска откройте в браузере: http://localhost:8000

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 1. Системные пакеты — если нет, устанавливаем (попросит пароль один раз)
MISSING_SYS=""
command -v ffmpeg &>/dev/null || MISSING_SYS="$MISSING_SYS ffmpeg"
ldconfig -p 2>/dev/null | grep -q "libatomic.so.1" || MISSING_SYS="$MISSING_SYS libatomic1"

if [ -n "$MISSING_SYS" ]; then
  echo "Устанавливаю системные пакеты:$MISSING_SYS (потребуется пароль)..."
  sudo apt-get update -qq && sudo apt-get install -y ffmpeg libatomic1 || \
    echo "ПРЕДУПРЕЖДЕНИЕ: не удалось установить системные пакеты (нет сети?), продолжаем..."
fi

# 2. Виртуальное окружение
cd "$ROOT/backend"
if [ ! -d "venv" ]; then
  echo "Создаю виртуальное окружение..."
  # Пробуем с --system-site-packages (позволяет использовать уже установленные системные пакеты)
  if ! python3 -m venv --system-site-packages venv 2>/dev/null; then
    if ! python3 -m venv venv 2>/dev/null; then
      echo "Ошибка: установите python3-venv: sudo apt install -y python3-venv"
      exit 1
    fi
  fi
fi

# 3. Зависимости Python
echo "Проверяю/устанавливаю зависимости Python..."

# Обновляем pip — игнорируем ошибки сети
./venv/bin/pip install -q --upgrade pip 2>/dev/null || true

# Устанавливаем требования; при ошибке сети — продолжаем без прерывания
PIP_OUTPUT=$(./venv/bin/pip install -q -r requirements.txt 2>&1)
PIP_EXIT=$?
if [ $PIP_EXIT -ne 0 ]; then
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  ВНИМАНИЕ: Не удалось установить некоторые пакеты."
  echo "  Возможно, нет подключения к интернету."
  echo ""
  echo "  Для полной установки подключите интернет и повторите:"
  echo "    bash start.sh"
  echo ""
  echo "  Проверяю наличие ключевых пакетов..."
  echo "════════════════════════════════════════════════════════"

  CORE_OK=true
  for pkg in fastapi uvicorn numpy scipy soundfile; do
    if ./venv/bin/python -c "import $pkg" 2>/dev/null; then
      echo "  ✓ $pkg"
    else
      echo "  ✗ $pkg — ОТСУТСТВУЕТ (критично)"
      CORE_OK=false
    fi
  done

  # Показываем статус необязательных пакетов
  for pkg in sqlalchemy passlib jose pedalboard pyloudnorm; do
    if ./venv/bin/python -c "import $pkg" 2>/dev/null; then
      echo "  ✓ $pkg"
    else
      echo "  ~ $pkg — не установлен (авторизация/часть функций недоступны)"
    fi
  done
  echo ""

  if [ "$CORE_OK" = false ]; then
    echo "ОШИБКА: Критические пакеты отсутствуют. Подключите интернет и запустите снова."
    exit 1
  fi
  echo "Основные пакеты найдены. Запускаю сервер (авторизация может быть недоступна)."
fi

# 4. Режим отладки (PRO без подписок)
# Если в корне проекта есть файл .debug — включаем MAGIC_MASTER_DEBUG=1
if [ -f "$ROOT/.debug" ]; then
  export MAGIC_MASTER_DEBUG=1
  echo "[Режим отладки включён: .debug найден]"
fi

# 5. Запуск
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Сервер запущен! Откройте в браузере:"
echo "  http://localhost:8000"
echo "  Остановка: Ctrl+C"
echo "════════════════════════════════════════════════════════"
echo ""
exec ./venv/bin/python run.py
