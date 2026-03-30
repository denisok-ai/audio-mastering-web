# Развёртывание Magic Master на веб-сервере

Пошаговая инструкция по размещению приложения на сервере (Ubuntu 22.04 / 24.04), чтобы им можно было пользоваться по интернету, а не только локально.

**Рекомендуемый способ на выделенном сервере — без Docker:** systemd + Nginx дают меньшие накладные расходы и проще диагностику. Скрипт `deploy/deploy.sh` автоматизирует установку и обновление (см. раздел «Автоматический деплой (deploy.sh)» ниже).

---

## Что понадобится

- Сервер с **Ubuntu 22.04** или **24.04** (VPS или свой сервер).
- Доступ по SSH (логин и пароль или ключ).
- Доменное имя (опционально; без него можно открывать по IP и порту).

---

## Шаг 1. Подготовка архива на своей машине

1. Перейдите в каталог проекта:
   ```bash
   cd /путь/к/audio-mastering-web
   ```

2. Сделайте скрипт упаковки исполняемым и запустите его:
   ```bash
   chmod +x pack_for_deploy.sh
   ./pack_for_deploy.sh
   ```

3. В каталоге появится файл `audio-mastering-web-deploy-YYYYMMDD.tar.gz` — его нужно скопировать на сервер.

---

## Шаг 2. Копирование проекта на сервер

**Вариант А — через SCP (с вашего компьютера):**
```bash
scp audio-mastering-web-deploy-*.tar.gz user@IP_СЕРВЕРА:/home/user/
```
Подставьте `user` (логин) и `IP_СЕРВЕРА` (или домен).

**Вариант Б — через Git (если проект в репозитории):**
На сервере:
```bash
git clone https://ваш-репозиторий/audio-mastering-web.git
cd audio-mastering-web
```
Дальнейшие шаги те же, но без распаковки архива.

---

## Шаг 3. Подключение к серверу и распаковка

1. Подключитесь по SSH:
   ```bash
   ssh user@IP_СЕРВЕРА
   ```

2. Распакуйте архив в отдельную папку (подставьте свой путь к архиву):
   ```bash
   mkdir -p ~/audio-mastering-web
   cd ~/audio-mastering-web
   tar -xzf ~/audio-mastering-web-deploy-*.tar.gz
   ```
   После распаковки в каталоге должны появиться папки `backend`, `frontend`, `deploy` и файлы `start.sh`, `README.md`, `DEPLOY.md`.

3. Запомните полный путь к проекту — он понадобится для systemd. Например:
   ```bash
   pwd
   # например: /home/user/audio-mastering-web
   ```
   Дальше везде подставляйте свой путь вместо `/home/user/audio-mastering-web`.

---

## Шаг 4. Установка системных зависимостей на сервере

Выполните на сервере (потребуется пароль `sudo`):

```bash
sudo apt update
sudo apt install -y ffmpeg libatomic1 python3-venv python3-pip
```

При наличии Python 3.11 можно поставить `python3.11-venv` для лучшей производительности. Иначе подойдёт пакет `python3-venv` (обычно Python 3.10).

- **ffmpeg** — для загрузки и экспорта MP3/FLAC.
- **libatomic1** — для библиотеки обработки звука.
- **python3-venv** (или **python3.11-venv**) — для виртуального окружения Python.

---

## Шаг 5. Создание виртуального окружения и установка Python-зависимостей

1. Перейдите в каталог backend:
   ```bash
   cd /home/user/audio-mastering-web/backend
   # или: cd /home/user/backend  — если распаковалось без корневой папки
   ```

2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python3 -m venv venv
   ./venv/bin/pip install --upgrade pip
   ./venv/bin/pip install -r requirements.txt
   ```

3. Проверьте запуск (временно):
   ```bash
   ./venv/bin/python run_production.py
   ```
   В браузере откройте `http://IP_СЕРВЕРА:8000`. Если страница открылась — остановите сервер (**Ctrl+C**) и переходите к следующему шагу.

---

## Шаг 6. Запуск через systemd (постоянная работа в фоне)

Чтобы приложение работало после выхода из SSH и перезагрузки сервера:

1. Скопируйте шаблон сервиса:
   ```bash
   sudo cp /home/user/audio-mastering-web/deploy/systemd/magic-master.service /etc/systemd/system/
   ```

2. Откройте файл и замените путь на реальный:
   ```bash
   sudo nano /etc/systemd/system/magic-master.service
   ```
   Замените **все** вхождения `/path/to/audio-mastering-web` на ваш путь, например `/home/user/audio-mastering-web`.  
   При необходимости смените `User=www-data` на вашего пользователя (например `User=user`).

3. Включите и запустите сервис:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable magic-master
   sudo systemctl start magic-master
   sudo systemctl status magic-master
   ```

4. Проверка: откройте в браузере `http://IP_СЕРВЕРА:8000`.

**Полезные команды:**
- Остановить: `sudo systemctl stop magic-master`
- Перезапустить: `sudo systemctl restart magic-master`
- Логи: `sudo journalctl -u magic-master -f`

---

## Шаг 7. Открытие порта в файрволе (если включён ufw)

Если на сервере используется `ufw`, разрешите порт 8000:

```bash
sudo ufw allow 8000/tcp
sudo ufw reload
```

После этого приложение доступно по адресу: **http://IP_СЕРВЕРА:8000**.

---

## Домен magicmaster.pro и SSL

Пошаговая инструкция (DNS, certbot, `.env`, проверки): **[doc/MAGICMASTER_PRO.md](doc/MAGICMASTER_PRO.md)**.  
Шаблон nginx под этот домен: [deploy/nginx/magic-master.conf](deploy/nginx/magic-master.conf).

## Шаг 8 (опционально). Nginx и HTTPS

Если нужен доступ по домену и HTTPS (например, `https://master.example.com`):

1. Установите nginx и certbot:
   ```bash
   sudo apt install -y nginx certbot python3-certbot-nginx
   ```

2. Скопируйте конфиг и include-файл проксирования, затем отредактируйте домен:
   ```bash
   sudo cp /home/user/audio-mastering-web/deploy/nginx/magic-master.conf /etc/nginx/sites-available/
   sudo cp /home/user/audio-mastering-web/deploy/nginx/magic-master-proxy.inc /etc/nginx/sites-available/
   sudo nano /etc/nginx/sites-available/magic-master.conf
   ```
   Замените `your-domain.com` на ваш домен (в блоках server_name для HTTPS).

3. Включите сайт и проверьте nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/magic-master.conf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

4. Получите бесплатный SSL-сертификат:
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```
   Certbot сам настроит HTTPS в nginx.

5. В конфиге nginx должен быть проксирование на `http://127.0.0.1:8000` (как в шаблоне). После certbot сайт будет доступен по **https://your-domain.com**.

---

## Production: безопасность (CORS и webhook)

Перед выходом в production задайте на сервере:

- **MAGIC_MASTER_CORS_ORIGINS** — список разрешённых доменов через запятую (например `https://your-domain.com`). Пустое значение = разрешены все origins (`*`), что нежелательно в проде.
- **MAGIC_MASTER_YOOKASSA_WEBHOOK_IP_WHITELIST** — IP-адреса серверов YooKassa через запятую; webhook будет приниматься только с этих IP. Уточните актуальные IP в документации YooKassa.

Подробнее: [doc/RUNBOOK.md](doc/RUNBOOK.md), [.env.example](.env.example).

---

## Переменные окружения (опционально)

Их можно задать в systemd-сервисе (секция `[Service]`):

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MAGIC_MASTER_HOST` | Адрес прослушивания | 0.0.0.0 |
| `MAGIC_MASTER_PORT` | Порт | 8000 |
| `MAGIC_MASTER_MAX_UPLOAD_MB` | Макс. размер загрузки (МБ) | 100 |
| `MAGIC_MASTER_TEMP_DIR` | Каталог временных файлов | /tmp/masterflow |

Пример в `magic-master.service`:
```ini
Environment="MAGIC_MASTER_PORT=8000"
Environment="MAGIC_MASTER_MAX_UPLOAD_MB=100"
```

---

## Краткая шпаргалка

| Действие | Команда |
|----------|---------|
| Собрать архив для выгрузки | `./pack_for_deploy.sh` |
| Запустить сервис | `sudo systemctl start magic-master` |
| Остановить | `sudo systemctl stop magic-master` |
| Перезапустить | `sudo systemctl restart magic-master` |
| Смотреть логи | `sudo journalctl -u magic-master -f` |
| Проверить состояние | `curl http://IP_СЕРВЕРА:8000/api/health` или откройте /status |
| Открыть приложение | http://IP_СЕРВЕРА:8000 или https://ваш-домен |

**Действия при сбоях:** см. [doc/RUNBOOK.md](doc/RUNBOOK.md) (health, логи, бэкап БД, перезапуск).

---

## Автоматический бэкап БД

Чтобы регулярно сохранять копию базы данных, используйте скрипт `deploy/backup_db.sh` по cron.

1. На сервере перейдите в каталог проекта и сделайте скрипт исполняемым:
   ```bash
   cd /home/user/audio-mastering-web
   chmod +x deploy/backup_db.sh
   ```

2. Укажите в скрипте путь к backend (переменная `BACKEND_DIR`) и каталог для бэкапов (`BACKUP_DIR`). По умолчанию бэкапы сохраняются в **`backups_db/`** в корне проекта (каталог в `.gitignore`). Отдельно на VPS иногда создают **`backups/`** для полных tar-архивов по cron — он тоже игнорируется Git.

3. Добавьте задачу в cron (например, раз в день в 3:00):
   ```bash
   crontab -e
   # добавить строку (подставьте свой путь):
   0 3 * * * /home/user/audio-mastering-web/deploy/backup_db.sh
   ```

Скрипт создаёт файл вида `magic_master_YYYYMMDD_HHMMSS.sqlite3`. Старые бэкапы рекомендуется удалять или выносить в облако (скрипт можно доработать под себя).

После выполнения шагов 1–7 приложение доступно в веб по адресу сервера и порту 8000; при необходимости шаг 8 добавляет домен и HTTPS.

---

## Выделенный сервер (6 vCPU / 12 GB RAM / 160 GB SSD)

Для сервера **только под Magic Master** (например 6 vCPU, 12 GB RAM, Ubuntu 22.04) рекомендуется деплой **без Docker**: systemd + Nginx (меньше расход памяти, проще диагностика).

### Автоматический деплой (deploy.sh) — без Docker

Скрипт `deploy/deploy.sh` устанавливает зависимости, создаёт пользователя `magicmaster`, venv, systemd-сервис и конфиг Nginx.

**Первый запуск (полная установка):**

```bash
# Клонирование в /opt/magic-master
sudo mkdir -p /opt/magic-master
cd /opt/magic-master
sudo git clone https://github.com/denisok-ai/audio-mastering-web.git .
# Или скопируйте проект в /opt/magic-master (backend, frontend, deploy, .env.example)

cd deploy
chmod +x deploy.sh
sudo INSTALL_DIR=/opt/magic-master ./deploy.sh install
```

После установки создайте/отредактируйте `/opt/magic-master/.env` (JWT_SECRET, MAGIC_MASTER_ADMIN_EMAIL и т.д.) и перезапустите: `sudo systemctl restart magic-master`.

**Обновление кода:**

```bash
cd /opt/magic-master
sudo git pull
cd deploy && sudo ./deploy.sh update
```

Или из каталога репозитория на другой машине (обновление через rsync):  
`sudo INSTALL_DIR=/opt/magic-master ./deploy.sh update` (при необходимости задайте PROJECT_ROOT).

**Отклонения на сервере:** после `git pull` рабочее дерево должно быть чистым; конфиги systemd/nginx/journald лежат **вне** Git. Чек-лист и типичные конфликты — **[doc/PRODUCTION_DRIFT.md](doc/PRODUCTION_DRIFT.md)**. Логи мастеринга (`MAGIC_MASTER_MASTERING_TRACE`) и регрессия качества — **[doc/MASTERING_REGRESSION.md](doc/MASTERING_REGRESSION.md)**.

### Рекомендуемые переменные окружения

Скопируйте в `.env` значения из `deploy/env.production-medium.example` (или добавьте в существующий `.env`):

- `MAGIC_MASTER_MAX_UPLOAD_MB=250` — крупные файлы (на 160 GB SSD допустимо)
- `MAGIC_MASTER_SEMAPHORE_PRIORITY=4` — до 4 одновременных мастерингов для Pro/Studio
- `MAGIC_MASTER_SEMAPHORE_NORMAL=2` — до 2 для Free
- `MAGIC_MASTER_JOBS_MAX_ENTRIES=300` — больше задач в очереди
- `MAGIC_MASTER_JOBS_DONE_TTL_SECONDS=7200` — дольше хранить результаты (2 часа)
- `MAGIC_MASTER_GLOBAL_RATE_LIMIT=600` — выше лимит запросов в минуту

Остальное (JWT, админ, SMTP, YooKassa и т.д.) — как в `.env.example`.

### Альтернатива: развёртывание через Docker

Если нужен именно Docker (например, общая инфраструктура уже в контейнерах):

```bash
apt-get update && apt-get install -y git docker.io
systemctl enable docker && systemctl start docker
cd /root
git clone https://github.com/denisok-ai/audio-mastering-web.git
cd audio-mastering-web
cp .env.example .env
nano .env   # задать JWT_SECRET, ADMIN_EMAIL, ADMIN_PASSWORD и др.

docker build -t magic-master:latest .
docker run -d --name magic-master -p 8000:8000 \
  --restart=unless-stopped \
  -e MAGIC_MASTER_HOST=0.0.0.0 \
  -v $(pwd)/.env:/app/backend/.env:ro \
  -v $(pwd)/PROGRESS.md:/app/PROGRESS.md:ro \
  magic-master:latest python run_production.py
```

Проверка: `curl http://localhost:8000/api/health`. Перед Nginx увеличьте `client_max_body_size` (например до 260M) и таймауты (`proxy_read_timeout 300s`).
