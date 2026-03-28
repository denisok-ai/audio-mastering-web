#!/usr/bin/env bash
# Magic Master — мониторинг дискового пространства.
# Проверяет свободное место, очищает tmp/журналы при нехватке, шлёт алерт.
# Cron: */15 * * * * /opt/magic-master/deploy/disk_monitor.sh
#
# RESERVE_PERCENT — порог свободного места (по умолчанию 15)
# ALERT_SCRIPT    — путь к скрипту алертов

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESERVE_PERCENT="${RESERVE_PERCENT:-15}"
ALERT_SCRIPT="${ALERT_SCRIPT:-$SCRIPT_DIR/send_alert.sh}"
LOCKFILE="/tmp/disk_monitor_alert.lock"
TEMP_DIR="${MAGIC_MASTER_TEMP_DIR:-/tmp/masterflow}"

get_disk_free_percent() {
    local used
    used=$(df / --output=pcent 2>/dev/null | tail -1 | tr -d ' %')
    echo $((100 - used))
}

send_alert() {
    if [ -x "$ALERT_SCRIPT" ]; then
        "$ALERT_SCRIPT" "DISK ALERT" "$1" 2>/dev/null || true
    fi
}

FREE=$(get_disk_free_percent)

if [ "$FREE" -ge "$RESERVE_PERCENT" ]; then
    rm -f "$LOCKFILE"
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Свободно ${FREE}% (порог ${RESERVE_PERCENT}%). Запуск очистки..."

# 1. Очистка временных файлов мастеринга старше 2 часов
if [ -d "$TEMP_DIR" ]; then
    find "$TEMP_DIR" -type f -mmin +120 -delete 2>/dev/null || true
    find "$TEMP_DIR" -type d -empty -delete 2>/dev/null || true
    echo "  Очищен $TEMP_DIR (файлы старше 2ч)"
fi

# 2. Сжатие журнала systemd
if command -v journalctl >/dev/null 2>&1; then
    journalctl --vacuum-size=200M 2>/dev/null || true
    echo "  journalctl vacuum до 200M"
fi

# 3. Очистка apt-кеша
if command -v apt-get >/dev/null 2>&1; then
    apt-get clean 2>/dev/null || true
    echo "  apt-get clean"
fi

# 4. Пересчитываем
FREE=$(get_disk_free_percent)
echo "  После очистки: свободно ${FREE}%"

if [ "$FREE" -lt "$RESERVE_PERCENT" ]; then
    if [ ! -f "$LOCKFILE" ] || [ "$(find "$LOCKFILE" -mmin +60 2>/dev/null)" ]; then
        MSG="ВНИМАНИЕ: свободно ${FREE}% дискового пространства (порог ${RESERVE_PERCENT}%). Автоочистка не помогла. Сервер: $(hostname)"
        send_alert "$MSG"
        touch "$LOCKFILE"
        echo "  АЛЕРТ отправлен: $MSG"
    else
        echo "  Алерт уже отправлен менее часа назад, повторно не шлём"
    fi
fi
