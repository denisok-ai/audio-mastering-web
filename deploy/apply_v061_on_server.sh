#!/usr/bin/env bash
# Magic Master v0.6.1 — деплой AI-консультанта + админ-панель бота + аномалии.
# Запуск: sshpass -p 'PASSWORD' ssh root@91.218.244.146 'bash -s' < deploy/apply_v061_on_server.sh
set -euo pipefail

APP_DIR="/opt/magic-master"
VENV="$APP_DIR/venv"

echo "=== [1/7] Git pull ==="
cd "$APP_DIR"
git fetch --all
git reset --hard origin/main
echo "OK: код обновлён до $(git log --oneline -1)"

echo "=== [2/7] venv + зависимости ==="
if [ ! -f "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    echo "venv создан"
fi
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$APP_DIR/backend/requirements.txt" -q
echo "OK: зависимости установлены (psutil, aiogram, openai и пр.)"

echo "=== [3/7] .env — дополнение недостающих ключей ==="
if [ -f "$APP_DIR/.env" ]; then
    # AI backend
    if ! grep -q '^MAGIC_MASTER_AI_BACKEND=' "$APP_DIR/.env" 2>/dev/null; then
        echo 'MAGIC_MASTER_AI_BACKEND=deepseek' >> "$APP_DIR/.env"
        echo "Добавлен AI_BACKEND=deepseek"
    fi
    # Anomaly / alert keys (закомментированы — включить вручную)
    if ! grep -q 'ANOMALY_CHECK_INTERVAL' "$APP_DIR/.env" 2>/dev/null; then
        cat >> "$APP_DIR/.env" <<'ENVBLOCK'

# ─── Мониторинг аномалий (v0.6.1) ────────────────────────────────────────────
# Включите для автоалертов в Telegram (TELEGRAM_BOT_TOKEN + ADMIN_CHAT_ID)
# MAGIC_MASTER_ALERT_MONITORING_ENABLED=1
# MAGIC_MASTER_ALERT_QUEUE_THRESHOLD=25
# MAGIC_MASTER_ANOMALY_CHECK_INTERVAL=60
# MAGIC_MASTER_ANOMALY_CPU_THRESHOLD=85
# MAGIC_MASTER_ANOMALY_RAM_THRESHOLD=85
# MAGIC_MASTER_ANOMALY_DISK_THRESHOLD=85
# MAGIC_MASTER_ANOMALY_ERROR_RATE_THRESHOLD=35
# MAGIC_MASTER_ANOMALY_RSS_MAX_MB=8192
ENVBLOCK
        echo "Добавлены ключи мониторинга аномалий (закомментированы)"
    fi
    # User bot keys
    if ! grep -q 'USER_BOT_TOKEN' "$APP_DIR/.env" 2>/dev/null; then
        cat >> "$APP_DIR/.env" <<'ENVBLOCK'

# ─── Telegram user bot (v0.6.0) ──────────────────────────────────────────────
# MAGIC_MASTER_USER_BOT_TOKEN=
# MAGIC_MASTER_USER_BOT_WEBHOOK_SECRET=
# MAGIC_MASTER_USER_BOT_CHANNEL_ID=@PROmagicmaster
# MAGIC_MASTER_PUBLIC_BASE_URL=https://magicmaster.pro
ENVBLOCK
        echo "Добавлены ключи user bot (закомментированы)"
    fi
else
    cp "$APP_DIR/deploy/env.production" "$APP_DIR/.env"
    echo ".env создан из шаблона"
fi
echo "OK: .env настроен"

echo "=== [4/7] systemd ==="
cp "$APP_DIR/deploy/systemd/magic-master.service" /etc/systemd/system/magic-master.service
cp "$APP_DIR/deploy/systemd/magic-master-alert@.service" /etc/systemd/system/magic-master-alert@.service 2>/dev/null || true
systemctl daemon-reload
echo "OK: systemd обновлён"

echo "=== [5/7] Nginx ==="
cp "$APP_DIR/deploy/nginx/magic-master-proxy.inc" /etc/nginx/sites-available/magic-master-proxy.inc
nginx -t && systemctl reload nginx
echo "OK: nginx обновлён"

echo "=== [6/7] Права + перезапуск ==="
chown -R magicmaster:magicmaster "$APP_DIR" 2>/dev/null || true
systemctl restart magic-master
echo "Ждём запуск..."
sleep 4

if systemctl is-active --quiet magic-master; then
    echo "OK: magic-master запущен"
else
    echo "ОШИБКА: magic-master не запустился!"
    journalctl -u magic-master --no-pager -n 30
    exit 1
fi

echo "=== [7/7] Проверки ==="
# Health
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/health 2>/dev/null || echo "000")
echo "Health check: HTTP $HTTP_CODE"

# Version
VERSION_JSON=$(curl -s http://127.0.0.1:8000/api/version 2>/dev/null || echo '{}')
echo "Version: $VERSION_JSON"

# Внешний HTTPS
EXT_CODE=$(curl -sk -o /dev/null -w '%{http_code}' https://magicmaster.pro/api/health 2>/dev/null || echo "000")
echo "External HTTPS health: HTTP $EXT_CODE"

echo ""
echo "=== Деплой v0.6.1 завершён ==="
echo "Новое:"
echo "  - AI-консультант (DeepSeek, двуязычная KB)"
echo "  - Telegram user bot (команды, мастеринг, AI-чат)"
echo "  - Админ-панель бота: /admin (8 кнопок), /server, /jobs, /errors, /report"
echo "  - Фоновый мониторинг аномалий (включить: ALERT_MONITORING_ENABLED=1)"
echo ""
echo "Для включения user bot: раскомментируйте USER_BOT_* в /opt/magic-master/.env"
echo "Для алертов: раскомментируйте ALERT_MONITORING_ENABLED=1 в .env"
