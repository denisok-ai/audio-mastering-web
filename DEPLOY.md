# Развёртывание Magic Master на веб-сервере

Пошаговая инструкция по размещению приложения на сервере (Ubuntu 22.04 / 24.04), чтобы им можно было пользоваться по интернету, а не только локально.

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
sudo apt install -y ffmpeg libatomic1 python3.10-venv python3-pip
```

- **ffmpeg** — для загрузки и экспорта MP3/FLAC.
- **libatomic1** — для библиотеки обработки звука.
- **python3.10-venv** — для виртуального окружения Python.

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

## Шаг 8 (опционально). Nginx и HTTPS

Если нужен доступ по домену и HTTPS (например, `https://master.example.com`):

1. Установите nginx и certbot:
   ```bash
   sudo apt install -y nginx certbot python3-certbot-nginx
   ```

2. Скопируйте конфиг и отредактируйте домен:
   ```bash
   sudo cp /home/user/audio-mastering-web/deploy/nginx/magic-master.conf /etc/nginx/sites-available/
   sudo nano /etc/nginx/sites-available/magic-master.conf
   ```
   Замените `your-domain.com` на ваш домен.

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

2. Укажите в скрипте путь к backend (переменная `BACKEND_DIR`) и каталог для бэкапов (`BACKUP_DIR`). По умолчанию бэкапы сохраняются в `./backups_db` в корне проекта.

3. Добавьте задачу в cron (например, раз в день в 3:00):
   ```bash
   crontab -e
   # добавить строку (подставьте свой путь):
   0 3 * * * /home/user/audio-mastering-web/deploy/backup_db.sh
   ```

Скрипт создаёт файл вида `magic_master_YYYYMMDD_HHMMSS.sqlite3`. Старые бэкапы рекомендуется удалять или выносить в облако (скрипт можно доработать под себя).

После выполнения шагов 1–7 приложение доступно в веб по адресу сервера и порту 8000; при необходимости шаг 8 добавляет домен и HTTPS.
