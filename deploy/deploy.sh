#!/usr/bin/env bash
# Magic Master — развёртывание и обновление на сервере (без Docker).
# Использование:
#   Первый запуск (полная установка):  sudo ./deploy.sh install
#   Обновление кода:                    sudo ./deploy.sh update
#   Только перезапуск сервиса:          sudo ./deploy.sh restart
#
# Ожидается: репозиторий уже склонирован в INSTALL_DIR (или склонируйте вручную перед install).

set -e

INSTALL_DIR="${INSTALL_DIR:-/opt/magic-master}"
APP_USER="${APP_USER:-magicmaster}"
APP_GROUP="${APP_GROUP:-magicmaster}"
BACKEND_DIR="$INSTALL_DIR/backend"
VENV_DIR="$INSTALL_DIR/venv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Определяем, откуда брать файлы: из PROJECT_ROOT (если запускаем из репо) или из INSTALL_DIR
if [ -d "$PROJECT_ROOT/backend" ] && [ -f "$PROJECT_ROOT/backend/requirements.txt" ]; then
    SOURCE_ROOT="$PROJECT_ROOT"
else
    SOURCE_ROOT="$INSTALL_DIR"
fi

log() { echo "[deploy] $*"; }

# ─── install: полная установка ─────────────────────────────────────────────────
do_install() {
    log "Полная установка в $INSTALL_DIR"

    if [ "$(id -u)" -ne 0 ]; then
        echo "Запустите с sudo для установки: sudo $0 install" >&2
        exit 1
    fi

    # Системные пакеты (всегда ставим python3-venv и ffmpeg; для python3.11 — свой venv-пакет)
    log "Установка системных пакетов (ffmpeg, python3-venv, nginx...)"
    apt-get update -qq
    apt-get install -y -qq ffmpeg libatomic1 python3-venv python3-pip
    if command -v python3.11 &>/dev/null; then
        apt-get install -y -qq python3.11-venv 2>/dev/null || true
    fi
    if command -v python3.10 &>/dev/null && ! command -v python3.11 &>/dev/null; then
        apt-get install -y -qq python3.10-venv 2>/dev/null || true
    fi
    apt-get install -y -qq nginx certbot python3-certbot-nginx 2>/dev/null || true

    # Каталог приложения
    mkdir -p "$INSTALL_DIR"
    if [ ! -d "$INSTALL_DIR/backend" ]; then
        if [ -d "$PROJECT_ROOT/backend" ]; then
            log "Копирование проекта из $PROJECT_ROOT в $INSTALL_DIR"
            cp -a "$PROJECT_ROOT/backend" "$PROJECT_ROOT/frontend" "$PROJECT_ROOT/deploy" "$INSTALL_DIR/" 2>/dev/null || true
            [ -f "$PROJECT_ROOT/PROGRESS.md" ] && cp "$PROJECT_ROOT/PROGRESS.md" "$INSTALL_DIR/" || true
            [ -f "$PROJECT_ROOT/.env.example" ] && cp "$PROJECT_ROOT/.env.example" "$INSTALL_DIR/" || true
        else
            log "Ошибка: не найден backend в $PROJECT_ROOT. Сначала выполните: git clone ... $INSTALL_DIR && cd $INSTALL_DIR" >&2
            exit 1
        fi
    fi

    # Пользователь
    if ! getent group "$APP_GROUP" >/dev/null; then
        log "Создание группы $APP_GROUP"
        groupadd --system "$APP_GROUP"
    fi
    if ! getent passwd "$APP_USER" >/dev/null; then
        log "Создание пользователя $APP_USER"
        useradd --system --gid "$APP_GROUP" --home-dir "$INSTALL_DIR" --no-create-home "$APP_USER"
    fi

    # venv: создать или пересоздать, если нет pip
    if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/pip" ]; then
        if [ -d "$VENV_DIR" ]; then
            log "Удаление битого/пустого venv и пересоздание"
            rm -rf "$VENV_DIR"
        fi
        log "Создание venv в $VENV_DIR"
        python3 -m venv "$VENV_DIR" || { echo "Ошибка: установите пакет python3-venv (или python3.11-venv): apt-get install python3-venv" >&2; exit 1; }
        "$VENV_DIR/bin/pip" install --upgrade pip
    fi

    "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt" -q

    # .env
    if [ ! -f "$INSTALL_DIR/.env" ]; then
        if [ -f "$INSTALL_DIR/.env.example" ]; then
            cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
            log "Создан $INSTALL_DIR/.env из .env.example — отредактируйте его (JWT, ADMIN_EMAIL и т.д.)"
        else
            log "Файл .env не найден. Создайте его вручную в $INSTALL_DIR/.env"
        fi
    fi

    # Права
    chown -R "$APP_USER:$APP_GROUP" "$INSTALL_DIR"

    # systemd (конфиги берём из каталога приложения)
    if [ -f "$INSTALL_DIR/deploy/systemd/magic-master.service" ]; then
        sed "s|/opt/magic-master|$INSTALL_DIR|g; s|magicmaster|$APP_USER|g; s|magicmaster|$APP_GROUP|g" \
            "$INSTALL_DIR/deploy/systemd/magic-master.service" > /etc/systemd/system/magic-master.service
        systemctl daemon-reload
        systemctl enable magic-master
        systemctl start magic-master
        log "systemd: magic-master включён и запущен"
    fi

    # nginx (только если каталог есть — nginx установлен; иначе создаём каталоги и пробуем)
    if [ -f "$INSTALL_DIR/deploy/nginx/magic-master.conf" ]; then
        if [ ! -d /etc/nginx/sites-available ]; then
            log "Каталог /etc/nginx/sites-available отсутствует. Установите nginx: apt-get install -y nginx"
            mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled 2>/dev/null || true
        fi
        if [ -d /etc/nginx/sites-available ]; then
            mkdir -p /var/www/certbot
            cp "$INSTALL_DIR/deploy/nginx/magic-master.conf" /etc/nginx/sites-available/magic-master
            [ -f "$INSTALL_DIR/deploy/nginx/magic-master-proxy.inc" ] && \
                cp "$INSTALL_DIR/deploy/nginx/magic-master-proxy.inc" /etc/nginx/sites-available/
            ln -sf /etc/nginx/sites-available/magic-master /etc/nginx/sites-enabled/magic-master
            rm -f /etc/nginx/sites-enabled/default
            if command -v nginx &>/dev/null; then
                nginx -t 2>/dev/null && systemctl reload nginx 2>/dev/null && log "nginx: конфиг установлен и перезагружен" || log "nginx: конфиг скопирован, перезагрузите nginx вручную после установки"
            else
                log "nginx не установлен; конфиг скопирован в /etc/nginx/sites-available/. После установки nginx выполните: ln -sf /etc/nginx/sites-available/magic-master /etc/nginx/sites-enabled/ && systemctl reload nginx"
            fi
        fi
    fi

    log "Установка завершена. Проверка: curl http://127.0.0.1:8000/api/health"
}

# ─── update: обновление кода и перезапуск ───────────────────────────────────────
do_update() {
    log "Обновление приложения в $INSTALL_DIR"

    if [ -d "$INSTALL_DIR/.git" ]; then
        cd "$INSTALL_DIR"
        git pull --ff-only
        cd - >/dev/null
    else
        log "Не найден .git в $INSTALL_DIR — копирование из $PROJECT_ROOT"
        if [ -d "$PROJECT_ROOT/backend" ]; then
            rsync -a --exclude='.git' --exclude='venv' --exclude='__pycache__' --exclude='*.db' \
                "$PROJECT_ROOT/backend/" "$INSTALL_DIR/backend/" 2>/dev/null || true
            rsync -a --exclude='.git' "$PROJECT_ROOT/frontend/" "$INSTALL_DIR/frontend/" 2>/dev/null || true
            [ -f "$PROJECT_ROOT/deploy/nginx/magic-master-proxy.inc" ] && \
                cp "$PROJECT_ROOT/deploy/nginx/magic-master-proxy.inc" "$INSTALL_DIR/deploy/nginx/" 2>/dev/null || true
        fi
    fi

    "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt" -q
    systemctl restart magic-master
    log "Сервис magic-master перезапущен"
}

# ─── restart ───────────────────────────────────────────────────────────────────
do_restart() {
    systemctl restart magic-master
    log "magic-master перезапущен"
}

# ─── main ─────────────────────────────────────────────────────────────────────
case "${1:-update}" in
    install) do_install ;;
    update)  do_update ;;
    restart) do_restart ;;
    *)
        echo "Использование: $0 {install|update|restart}" >&2
        echo "  install — полная установка (sudo)" >&2
        echo "  update  — git pull + pip install + restart" >&2
        echo "  restart — перезапуск systemd-сервиса" >&2
        exit 1
        ;;
esac
