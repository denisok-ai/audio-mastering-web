# Runbook — действия при сбоях и эксплуатация Magic Master

Краткая шпаргалка для оператора: как проверить состояние сервиса, где смотреть логи, как сделать бэкап БД и перезапустить приложение.

---

## 1. Проверка состояния (health)

**Эндпоинт:** `GET /api/health`  
Пример: `curl https://ваш-домен/api/health` или откройте в браузере `https://ваш-домен/api/health`.

**Ответ содержит:**
- `status` — `ok` (всё в порядке), `degraded` (есть проблемы) или компоненты в ошибке.
- `components.database` — доступность БД (ok / error / unavailable).
- `components.disk` — свободное место в каталоге `temp_dir` (МБ); при `free_mb < 200` статус диска `low`.
- `components.ffmpeg` — наличие ffmpeg в PATH (ok / missing).
- `jobs.running` — число активных задач мастеринга; `jobs.total_cached` — всего записей в кэше задач.
- `uptime_since` — время старта процесса.
- `version` — версия приложения.

**Что делать:**
- `database.status !== "ok"` — проверить путь к SQLite (переменная окружения или путь по умолчанию), права на файл БД, место на диске.
- `disk.status === "low"` — освободить место (очистить `temp_dir`, старые логи, бэкапы).
- `ffmpeg === "missing"` — установить ffmpeg: `sudo apt install -y ffmpeg`.

Публичная страница статуса с автообновлением: **https://ваш-домен/status**.

**Метрики для скрапинга (P58):** `GET /api/metrics` — плоский JSON: `uptime_seconds`, `jobs_running`, `jobs_total`, `version`. Удобно для внешних систем мониторинга и дашбордов.

---

## 2. Логи

Логи приложения пишутся в stdout/stderr и при запуске через systemd попадают в journald.

**Смотреть логи (последние и в реальном времени):**
```bash
sudo journalctl -u magic-master -f
```

**Последние N строк:**
```bash
sudo journalctl -u magic-master -n 200
```

**За последний час:**
```bash
sudo journalctl -u magic-master --since "1 hour ago"
```

При ошибках мастеринга или оплаты в логах ищут сообщения с уровнем ERROR (если включено логирование ошибок). Уведомления об ошибках мастеринга также отправляются в Telegram (если настроены `MAGIC_MASTER_TELEGRAM_*`).

---

## 3. Бэкап базы данных

**Через админку (вручную):** войдите в админ-панель как администратор → Настройки → кнопка «⬇ Backup DB». Скачается файл SQLite с текущим состоянием БД.

**Через API (если есть JWT админа):**
```bash
curl -H "Authorization: Bearer ВАШ_JWT" "https://ваш-домен/api/admin/backup/db" -o backup_$(date +%Y%m%d_%H%M).sqlite3
```

**Автоматический бэкап:** используйте скрипт `deploy/backup_db.sh` по cron (см. [DEPLOY.md](../DEPLOY.md#автоматический-бэкап-бд)).

Восстановление: остановить сервис, заменить файл БД на бэкап, запустить сервис. Путь к БД задаётся в конфигурации (см. `backend/app/database.py`, по умолчанию в каталоге backend).

---

## 4. Перезапуск сервиса

**Перезапуск приложения:**
```bash
sudo systemctl restart magic-master
```

**Проверить статус:**
```bash
sudo systemctl status magic-master
```

**Остановить / запустить:**
```bash
sudo systemctl stop magic-master
sudo systemctl start magic-master
```

После перезапуска активные задачи мастеринга (в памяти) теряются; пользователям нужно будет запустить мастеринг заново.

---

## 5. Production: CORS и webhook YooKassa

Перед выходом в production обязательно задайте на сервере:

1. **CORS** — список разрешённых доменов (иначе по умолчанию разрешены все `*`):
   ```bash
   # В .env или в systemd Environment=
   MAGIC_MASTER_CORS_ORIGINS=https://ваш-домен.com,https://www.ваш-домен.com
   ```

2. **Webhook YooKassa** — ограничение по IP (чтобы принимать уведомления только от серверов YooKassa):
   ```bash
   MAGIC_MASTER_YOOKASSA_WEBHOOK_IP_WHITELIST=IP1,IP2
   ```
   Актуальные IP уточняйте в документации YooKassa или у поддержки.

Подробнее: [.env.example](../.env.example), [DEPLOY.md](../DEPLOY.md).
