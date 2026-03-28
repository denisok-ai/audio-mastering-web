#!/usr/bin/env bash
# Magic Master — отправка email-алерта через msmtp.
# Использование: ./send_alert.sh "Тема" "Тело сообщения"
# Настройка: /etc/msmtprc (системный) или ~/.msmtprc (пользователь)
#
# ALERT_EMAIL — адрес получателя (по умолчанию dsosin@mail.ru)
# ALERT_FROM  — адрес отправителя (по умолчанию alert@magicmaster.pro)

set -euo pipefail

SUBJECT="${1:-Magic Master Alert}"
BODY="${2:-Событие без описания}"
ALERT_EMAIL="${ALERT_EMAIL:-dsosin@mail.ru}"
ALERT_FROM="${ALERT_FROM:-alert@magicmaster.pro}"
HOSTNAME_STR="$(hostname 2>/dev/null || echo unknown)"

if ! command -v msmtp >/dev/null 2>&1; then
    echo "msmtp не установлен. Установите: sudo apt install msmtp msmtp-mta" >&2
    exit 1
fi

printf "From: %s\nTo: %s\nSubject: [MagicMaster] %s\nContent-Type: text/plain; charset=utf-8\n\n%s\n\nСервер: %s\nВремя: %s\n" \
    "$ALERT_FROM" "$ALERT_EMAIL" "$SUBJECT" "$BODY" "$HOSTNAME_STR" "$(date '+%Y-%m-%d %H:%M:%S %Z')" \
    | msmtp "$ALERT_EMAIL"
