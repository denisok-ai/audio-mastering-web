#!/usr/bin/env bash
# Выполнить на прод-сервере от root (или с sudo).
# Обновляет код из git, ставит nginx+certbot, выпускает SSL (webroot), включает HTTPS-конфиг, правит CORS в .env.
#
# Обязательно: DNS A magicmaster.pro (и при необходимости www) → IP этого сервера.
#
#   export CERTBOT_EMAIL=you@example.com
#   bash /opt/magic-master/deploy/apply_magicmaster_pro_on_server.sh
#
# Или с локальной машины (где работает ssh root@91.218.244.146):
#   scp deploy/apply_magicmaster_pro_on_server.sh root@HOST:/tmp/
#   ssh root@HOST 'CERTBOT_EMAIL=you@example.com bash /tmp/apply_magicmaster_pro_on_server.sh'

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/magic-master}"
EMAIL="${CERTBOT_EMAIL:-}"

if [ -z "$EMAIL" ]; then
  echo "Задайте email для Let's Encrypt: export CERTBOT_EMAIL=your@email.com" >&2
  exit 1
fi

if [ ! -d "$INSTALL_DIR/backend" ] || [ ! -d "$INSTALL_DIR/deploy/nginx" ]; then
  echo "Не найден проект в INSTALL_DIR=$INSTALL_DIR (нужны backend/ и deploy/nginx/)." >&2
  exit 1
fi

echo "[1/7] Обновление кода из git..."
if [ -d "$INSTALL_DIR/.git" ]; then
  git -C "$INSTALL_DIR" fetch origin
  git -C "$INSTALL_DIR" reset --hard origin/main
else
  echo "  (нет .git — пропуск pull; скопируйте файлы вручную)"
fi

# После reset venv в корне не в .gitignore на старых клонах мог пропасть — пересоздать
if [ ! -x "$INSTALL_DIR/venv/bin/python" ]; then
  echo "[1b/7] Пересоздание venv (python3 -m venv)..."
  python3 -m venv "$INSTALL_DIR/venv"
  "$INSTALL_DIR/venv/bin/pip" install -q --upgrade pip
  "$INSTALL_DIR/venv/bin/pip" install -q -r "$INSTALL_DIR/backend/requirements.txt"
  chown -R magicmaster:magicmaster "$INSTALL_DIR/venv" 2>/dev/null || true
fi

echo "[2/7] Пакеты nginx, certbot..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx certbot python3-certbot-nginx

echo "[3/7] Каталог webroot, firewall..."
mkdir -p /var/www/certbot
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow 80/tcp || true
  ufw allow 443/tcp || true
  ufw reload || true
fi

LE_CERT="/etc/letsencrypt/live/magicmaster.pro/fullchain.pem"
NGX_SITE="/etc/nginx/sites-available/magic-master"
PROXY_INC="/etc/nginx/sites-available/magic-master-proxy.inc"

cp "$INSTALL_DIR/deploy/nginx/magic-master-proxy.inc" "$PROXY_INC"

if [ ! -f "$LE_CERT" ]; then
  echo "[4/7] Первый выпуск сертификата (HTTP bootstrap + certbot webroot)..."
  cp "$INSTALL_DIR/deploy/nginx/magic-master-http-bootstrap.conf" "$NGX_SITE"
  ln -sf "$NGX_SITE" /etc/nginx/sites-enabled/magic-master
  rm -f /etc/nginx/sites-enabled/default
  nginx -t
  systemctl reload nginx

  certbot certonly --webroot -w /var/www/certbot \
    -d magicmaster.pro -d www.magicmaster.pro \
    --non-interactive --agree-tos --email "$EMAIL" \
    --preferred-challenges http
else
  echo "[4/7] Сертификат уже есть: $LE_CERT"
fi

echo "[5/7] Полный конфиг HTTPS..."
cp "$INSTALL_DIR/deploy/nginx/magic-master.conf" "$NGX_SITE"
nginx -t
systemctl reload nginx

echo "[6/7] CORS в .env..."
ENV_FILE="$INSTALL_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  if grep -q '^MAGIC_MASTER_CORS_ORIGINS=' "$ENV_FILE"; then
    sed -i 's|^MAGIC_MASTER_CORS_ORIGINS=.*|MAGIC_MASTER_CORS_ORIGINS=https://magicmaster.pro,https://www.magicmaster.pro|' "$ENV_FILE"
  else
    echo 'MAGIC_MASTER_CORS_ORIGINS=https://magicmaster.pro,https://www.magicmaster.pro' >> "$ENV_FILE"
  fi
  chown magicmaster:magicmaster "$ENV_FILE" 2>/dev/null || true
else
  echo "  Нет $ENV_FILE — создайте из .env.example и повторите при необходимости."
fi

echo "[7/7] Перезапуск приложения..."
systemctl restart magic-master
sleep 2
systemctl is-active magic-master || true

echo ""
echo "Готово. Проверка:"
echo "  curl -sI https://magicmaster.pro/api/health"
echo "  curl -sI http://magicmaster.pro"
