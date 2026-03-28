#!/usr/bin/env bash
# Magic Master — полный бэкап: SQLite + .env + nginx конфиги.
# Gzip-сжатие, ротация с гарантией 15% свободного места на диске.
# Cron: 0 4,16 * * * /opt/magic-master/deploy/backup_full.sh
#
# Конфигурация через переменные окружения:
#   BACKUP_DIR        — каталог для бэкапов (по умолчанию /opt/magic-master/backups)
#   RESERVE_PERCENT   — процент свободного места, который нужно сохранить (по умолчанию 15)
#   ALERT_SCRIPT      — путь к скрипту алертов (по умолчанию deploy/send_alert.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
RESERVE_PERCENT="${RESERVE_PERCENT:-15}"
ALERT_SCRIPT="${ALERT_SCRIPT:-$SCRIPT_DIR/send_alert.sh}"
LOG_FILE="$BACKUP_DIR/backup.log"

DB_PATH="$PROJECT_ROOT/backend/magic_master.db"
ENV_PATH="$PROJECT_ROOT/.env"
NGINX_DIR="/etc/nginx"

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$BACKUP_DIR/backup_${STAMP}.tar.gz"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

send_alert() {
    if [ -x "$ALERT_SCRIPT" ]; then
        "$ALERT_SCRIPT" "BACKUP ALERT" "$1" 2>/dev/null || true
    fi
}

get_disk_usage_percent() {
    df "$BACKUP_DIR" --output=pcent 2>/dev/null | tail -1 | tr -d ' %'
}

get_disk_free_percent() {
    local used
    used=$(get_disk_usage_percent)
    echo $((100 - used))
}

rotate_old_backups() {
    local free
    free=$(get_disk_free_percent)
    if [ "$free" -ge "$RESERVE_PERCENT" ]; then
        return 0
    fi

    log "Свободно ${free}%, нужно ${RESERVE_PERCENT}%. Удаляю старые бэкапы..."
    local count=0
    while [ "$free" -lt "$RESERVE_PERCENT" ]; do
        local oldest
        oldest=$(find "$BACKUP_DIR" -name 'backup_*.tar.gz' -type f -printf '%T+ %p\n' 2>/dev/null \
                 | sort | head -1 | cut -d' ' -f2-)
        if [ -z "$oldest" ]; then
            log "ОШИБКА: нет старых бэкапов для удаления, но места недостаточно!"
            send_alert "Backup rotation: нет старых бэкапов для удаления, свободно ${free}% (нужно ${RESERVE_PERCENT}%)"
            return 1
        fi
        log "Удаляю: $oldest"
        rm -f "$oldest"
        count=$((count + 1))
        free=$(get_disk_free_percent)
    done
    log "Удалено $count старых бэкапов. Свободно: ${free}%"
}

mkdir -p "$BACKUP_DIR"

log "=== Начало бэкапа ==="

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

if [ -f "$DB_PATH" ]; then
    if command -v sqlite3 >/dev/null 2>&1; then
        sqlite3 "$DB_PATH" "VACUUM INTO '$TMPDIR/magic_master.sqlite3';"
        log "SQLite: скопирована"
    else
        cp "$DB_PATH" "$TMPDIR/magic_master.sqlite3"
        log "SQLite: скопирована (без VACUUM — sqlite3 не установлен)"
    fi
else
    log "WARN: БД не найдена: $DB_PATH"
fi

if [ -f "$ENV_PATH" ]; then
    cp "$ENV_PATH" "$TMPDIR/dot_env"
    log ".env: скопирован"
fi

if [ -d "$NGINX_DIR" ]; then
    mkdir -p "$TMPDIR/nginx"
    cp -r "$NGINX_DIR/sites-available" "$TMPDIR/nginx/" 2>/dev/null || true
    cp -r "$NGINX_DIR/sites-enabled" "$TMPDIR/nginx/" 2>/dev/null || true
    cp -r "$NGINX_DIR/conf.d" "$TMPDIR/nginx/" 2>/dev/null || true
    cp "$NGINX_DIR/nginx.conf" "$TMPDIR/nginx/" 2>/dev/null || true
    log "Nginx: конфиги скопированы"
fi

rotate_old_backups

tar -czf "$ARCHIVE" -C "$TMPDIR" .
ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
log "Архив создан: $ARCHIVE ($ARCHIVE_SIZE)"

FREE_NOW=$(get_disk_free_percent)
if [ "$FREE_NOW" -lt "$RESERVE_PERCENT" ]; then
    MSG="После бэкапа свободно ${FREE_NOW}% (порог ${RESERVE_PERCENT}%). Требуется внимание!"
    log "ALERT: $MSG"
    send_alert "$MSG"
fi

log "=== Бэкап завершён. Свободно: ${FREE_NOW}% ==="
