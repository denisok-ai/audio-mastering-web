#!/usr/bin/env bash
# Magic Master v0.5.7 — применение обновлений безопасности и инфраструктуры на сервере.
# Запуск: sshpass -p 'PASSWORD' ssh root@91.218.244.146 'bash -s' < deploy/apply_v057_on_server.sh
set -euo pipefail

APP_DIR="/opt/magic-master"
VENV="$APP_DIR/venv"

echo "=== [1/10] Git pull ==="
cd "$APP_DIR"
git fetch --all
git reset --hard origin/main
echo "OK: код обновлён"

echo "=== [2/10] venv ==="
if [ ! -f "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    echo "venv создан"
fi
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$APP_DIR/backend/requirements.txt" -q
echo "OK: зависимости установлены"

echo "=== [3/10] .env — production ==="
if [ -f "$APP_DIR/.env" ]; then
    # Обновляем только ключевые параметры, не перезаписывая пользовательские настройки
    # JWT_SECRET
    if grep -q 'change-me' "$APP_DIR/.env" 2>/dev/null; then
        JWT=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
        sed -i "s|^MAGIC_MASTER_JWT_SECRET=.*|MAGIC_MASTER_JWT_SECRET=$JWT|" "$APP_DIR/.env"
        echo "JWT_SECRET обновлён на случайный"
    fi
    # DEBUG=0
    if grep -q '^MAGIC_MASTER_DEBUG=1' "$APP_DIR/.env" 2>/dev/null; then
        sed -i 's|^MAGIC_MASTER_DEBUG=1|MAGIC_MASTER_DEBUG=0|' "$APP_DIR/.env"
        echo "DEBUG выключен"
    fi
    # Добавляем недостающие параметры из env.production
    for KEY in SEMAPHORE_PRIORITY SEMAPHORE_NORMAL JOBS_MAX_ENTRIES JOBS_DONE_TTL_SECONDS GLOBAL_RATE_LIMIT; do
        if ! grep -q "^MAGIC_MASTER_$KEY=" "$APP_DIR/.env" 2>/dev/null; then
            grep "^MAGIC_MASTER_$KEY=" "$APP_DIR/deploy/env.production" >> "$APP_DIR/.env" || true
            echo "Добавлен $KEY"
        fi
    done
    # CORS
    if ! grep -q '^MAGIC_MASTER_CORS_ORIGINS=' "$APP_DIR/.env" 2>/dev/null; then
        echo 'MAGIC_MASTER_CORS_ORIGINS=https://magicmaster.pro,https://www.magicmaster.pro' >> "$APP_DIR/.env"
        echo "CORS_ORIGINS добавлен"
    fi
else
    cp "$APP_DIR/deploy/env.production" "$APP_DIR/.env"
    echo ".env создан из шаблона"
fi
echo "OK: .env настроен"

echo "=== [4/10] msmtp ==="
if ! command -v msmtp >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq msmtp msmtp-mta
fi
if [ ! -f /etc/msmtprc ]; then
    cp "$APP_DIR/deploy/msmtprc.example" /etc/msmtprc
    chmod 600 /etc/msmtprc
    echo "ВНИМАНИЕ: /etc/msmtprc создан из шаблона — заполните SMTP-реквизиты!"
else
    echo "msmtp уже настроен"
fi
echo "OK: msmtp готов"

echo "=== [5/10] systemd ==="
cp "$APP_DIR/deploy/systemd/magic-master.service" /etc/systemd/system/magic-master.service
cp "$APP_DIR/deploy/systemd/magic-master-alert@.service" /etc/systemd/system/magic-master-alert@.service
systemctl daemon-reload
echo "OK: systemd обновлён"

echo "=== [6/10] Nginx ==="
cp "$APP_DIR/deploy/nginx/magic-master.conf" /etc/nginx/sites-available/magic-master
cp "$APP_DIR/deploy/nginx/magic-master-proxy.inc" /etc/nginx/sites-available/magic-master-proxy.inc
cp "$APP_DIR/deploy/nginx/gzip.conf" /etc/nginx/conf.d/gzip.conf
# Убираем конфликтующий gzip из nginx.conf если есть
if grep -q '^\s*gzip on;' /etc/nginx/nginx.conf 2>/dev/null; then
    sed -i 's/^\(\s*gzip on;\)/#\1 # moved to conf.d\/gzip.conf/' /etc/nginx/nginx.conf
    sed -i 's/^\(\s*gzip_types\)/#\1/' /etc/nginx/nginx.conf
fi
nginx -t && systemctl reload nginx
echo "OK: nginx обновлён"

echo "=== [7/10] Logrotate ==="
cp "$APP_DIR/deploy/logrotate/magic-master" /etc/logrotate.d/magic-master
echo "OK: logrotate настроен"

echo "=== [8/10] Journald ==="
mkdir -p /etc/systemd/journald.conf.d
cp "$APP_DIR/deploy/journald.conf.d/magic-master.conf" /etc/systemd/journald.conf.d/magic-master.conf
systemctl restart systemd-journald
echo "OK: journald ограничен (500M, 14d)"

echo "=== [9/10] Cron ==="
CRON_MARKER="# magic-master-v057"
if ! crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
    (crontab -l 2>/dev/null || true; cat <<EOF
# === Magic Master автоматизация === $CRON_MARKER
# Бэкап 2 раза в день (04:00 и 16:00)
0 4,16 * * * /opt/magic-master/deploy/backup_full.sh >> /opt/magic-master/backups/backup.log 2>&1
# Мониторинг диска каждые 15 минут
*/15 * * * * /opt/magic-master/deploy/disk_monitor.sh >> /var/log/disk_monitor.log 2>&1
# Certbot auto-renew (если установлен)
0 3 * * 1 certbot renew --quiet --deploy-hook "systemctl reload nginx" 2>/dev/null || true
EOF
    ) | crontab -
    echo "Cron-задачи добавлены"
else
    echo "Cron-задачи уже настроены"
fi
echo "OK: cron"

echo "=== [10/10] Перезапуск сервиса ==="
mkdir -p "$APP_DIR/backups"
chown -R magicmaster:magicmaster "$APP_DIR" 2>/dev/null || true
systemctl restart magic-master
sleep 3

if systemctl is-active --quiet magic-master; then
    echo "OK: magic-master запущен"
    # Проверка health
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/health 2>/dev/null || echo "000")
    echo "Health check: HTTP $HTTP_CODE"
else
    echo "ОШИБКА: magic-master не запустился!"
    journalctl -u magic-master --no-pager -n 20
    exit 1
fi

echo ""
echo "=== Обновление v0.5.7 завершено ==="
echo "Проверьте:"
echo "  - /etc/msmtprc — заполните SMTP-реквизиты для алертов"
echo "  - curl https://magicmaster.pro/api/version"
echo "  - curl https://magicmaster.pro/docs (должен вернуть 404)"
