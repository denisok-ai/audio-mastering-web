#!/usr/bin/env bash
# Автоматический бэкап SQLite БД Magic Master (VACUUM INTO).
# Запуск: из корня проекта или с указанием BACKEND_DIR.
# Cron: 0 3 * * * /path/to/audio-mastering-web/deploy/backup_db.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="${BACKEND_DIR:-$PROJECT_ROOT/backend}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups_db}"
DB_PATH="$BACKEND_DIR/magic_master.db"

if [ ! -f "$DB_PATH" ]; then
  echo "БД не найдена: $DB_PATH" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$BACKUP_DIR/magic_master_${STAMP}.sqlite3"

if command -v sqlite3 >/dev/null 2>&1; then
  sqlite3 "$DB_PATH" "VACUUM INTO '$OUT';"
  echo "Бэкап создан: $OUT"
else
  echo "sqlite3 не найден. Установите: sudo apt install sqlite3" >&2
  exit 1
fi
