#!/bin/bash
# Скрипт инициализации git и первого пуша на GitHub
# Запуск: bash push_to_github.sh YOUR_GITHUB_TOKEN
set -e
LOG="/tmp/git_push.log"
exec > >(tee "$LOG") 2>&1

TOKEN="${1:-}"
REPO="https://github.com/denisok-ai/audio-mastering-web.git"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Magic Master → GitHub push ==="
echo "Dir: $DIR"
echo "Repo: $REPO"
echo ""

cd "$DIR"

# 1. Git init (если ещё не)
if [ ! -d ".git" ]; then
  echo "[1/5] git init..."
  git init -b main
else
  echo "[1/5] Git уже инициализирован. Текущая ветка: $(git branch --show-current)"
fi

# 2. Git config (если не задан)
if [ -z "$(git config user.email)" ]; then
  git config user.email "denisok@users.noreply.github.com"
  git config user.name "denisok-ai"
  echo "    Config: имя/email заданы"
fi

# 3. .gitignore проверка
echo "[2/5] Проверка .gitignore..."
git check-ignore -v backend/venv 2>/dev/null && echo "    venv: игнорируется ✓" || true

# 4. Добавить все файлы
echo "[3/5] git add ..."
git add .
echo "    Статус после add:"
git status --short | head -30

# 5. Коммит
if git diff --cached --quiet; then
  echo "[4/5] Нет изменений для коммита — пропускаем"
else
  echo "[4/5] git commit..."
  git commit -m "$(cat <<'COMMITMSG'
feat: genre presets, A/B player, waveform visualization

- 6 genre style presets: Stream, EDM, Hip-Hop, Classical, Podcast, Lo-fi
- apply_style_eq(): per-genre 5-band EQ correction in mastering pipeline
- Visual style cards grid replacing preset dropdown
- A/B comparison player: switch original ↔ mastered waveform in browser
- Web Audio API waveform player with playhead, seek, scrub bar
- Before/After LUFS panel, pipeline step visualization
- localStorage history (last 8 sessions), toast notifications
- Page-wide drag & drop with fullscreen overlay
- Keyboard shortcuts: Space = play/pause, Enter = master
- Backend /api/styles endpoint, peak dBFS + duration in /api/measure
COMMITMSG
)"
fi

# 6. Remote
echo "[5/5] Настройка remote и push..."
if [ -n "$TOKEN" ]; then
  REMOTE_URL="https://${TOKEN}@github.com/denisok-ai/audio-mastering-web.git"
else
  REMOTE_URL="$REPO"
  echo "    ВНИМАНИЕ: токен не передан — push может потребовать ввода пароля"
fi

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REMOTE_URL"
  echo "    Remote обновлён"
else
  git remote add origin "$REMOTE_URL"
  echo "    Remote добавлен"
fi

git push -u origin main

echo ""
echo "=== ГОТОВО ==="
echo "Репозиторий: https://github.com/denisok-ai/audio-mastering-web"
echo "Лог сохранён: $LOG"
