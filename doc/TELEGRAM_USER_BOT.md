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
