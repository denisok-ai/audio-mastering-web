# Telegram user bot (Magic Master)

Отдельный бот для пользователей (не путать с `MAGIC_MASTER_TELEGRAM_BOT_TOKEN` для админ-уведомлений).

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `MAGIC_MASTER_USER_BOT_TOKEN` | Токен от @BotFather |
| `MAGIC_MASTER_USER_BOT_WEBHOOK_SECRET` | Секрет для заголовка `X-Telegram-Bot-Api-Secret-Token` |
| `MAGIC_MASTER_USER_BOT_CHANNEL_ID` | `@PROmagicmaster` или числовой id канала |
| `MAGIC_MASTER_PUBLIC_BASE_URL` | `https://magicmaster.pro` — webhook и ссылки |

## Настройка

1. Создать бота в @BotFather, включить **Groups** при необходимости.
2. Добавить бота админом канала @PROmagicmaster (если постите в канал).
3. Заполнить `.env`, перезапустить сервис. При старте вызывается `setWebhook` на `{PUBLIC_BASE_URL}/bot/webhook`.
4. Nginx: в репозитории уже есть `location /bot/` в `deploy/nginx/magic-master-proxy.inc` — скопировать на сервер и `nginx -t && reload`.

## Привязка аккаунта

Пользователь: `/link email@site.ru` → код на почту → `/code 123456`. Нужен настроенный SMTP.

## Cron (welcome + weekly DM)

Из каталога `backend`:

```bash
cd /opt/magic-master/backend && \
  PYTHONPATH=. /opt/magic-master/venv/bin/python -m app.bot.run_engagement
```

Рекомендуется 1 раз в день. Для поста статистики в канал раз в неделю можно вызывать из того же окружения:

```python
from app.bot.engagement import post_channel_weekly_stats
post_channel_weekly_stats()
```

## Лимиты Telegram

Скачивание файлов в боте до **20 МБ**; большие треки — через сайт.

## Админ-панель (русский интерфейс)

Доступ только пользователям с флагом `is_admin` в БД и привязанным Telegram (`/link`).

- **`/admin`** — меню с кнопками: Сервер, Здоровье, Статистика, Задачи, Пользователи, Выручка, Ошибки, Полный отчёт.
- **`/server`** — кратко: CPU, RAM, диск, load average.
- **`/stats`** — пользователи и мастеринги (агрегаты).
- **`/jobs`** — сколько задач в работе / в очереди, семафоры.
- **`/errors`** — последние ошибки мастеринга из кэша и SQLite.
- **`/report`** — полный операционный отчёт (несколько сообщений, если длинный).
- **`/broadcast`** — рассылка текстом всем с привязанным Telegram.

Фоновый **мониторинг аномалий** (CPU, RAM, диск, RSS процесса, доля ошибок мастеринга за час, порог очереди задач) запускается при **`MAGIC_MASTER_ALERT_MONITORING_ENABLED=1`** и шлёт алерты в **`MAGIC_MASTER_TELEGRAM_ADMIN_CHAT_ID`** (тот же бот, что и прочие админ-уведомления). Интервал и пороги: `MAGIC_MASTER_ANOMALY_CHECK_INTERVAL`, `ANOMALY_CPU_THRESHOLD`, `ANOMALY_RAM_THRESHOLD`, `ANOMALY_DISK_THRESHOLD`, `ANOMALY_ERROR_RATE_THRESHOLD`, `ANOMALY_RSS_MAX_MB`, `ANOMALY_MIN_JOBS_FOR_ERROR_RATE` (см. `config.py`).
