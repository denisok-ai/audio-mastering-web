# Аудит связей фронтенд ↔ бэкенд — Magic Master

Документ фиксирует соответствие API и фронтенда: какие эндпоинты кем используются, разрывы и рекомендации по доработкам.

> **Обновлено:** 2026-03. Источник: разбор `backend/app` (роутеры, main), `frontend/*.html`, `frontend/app.js`.

---

## 1. Сводка по роутерам

| Роутер | Префикс | Назначение |
|--------|---------|------------|
| main.py | — | Корневые эндпоинты: /api, /api/version, /api/health, /api/metrics, /api/locale; маршруты страниц (/, /app, /admin, …). |
| admin.py | /api/admin | Админка: пользователи, транзакции, новости, рассылки, настройки, промпты, отчёты, аудит, бэкап. |
| payments.py | /api/payments | Тарифы (GET /plans), создание платежа (POST /create), webhook YooKassa (POST /webhook). |
| misc.py | — | Публичные: /api/news, /api/debug-mode, /api/limits, /api/progress, /api/presets, /api/presets/community, /api/extensions, /api/styles, POST /api/measure. |
| ai_router.py | — | AI: /api/ai/limits, POST /api/ai/recommend, report, nl-config, chat. |
| auth.py | — | Авторизация и профиль: register, login, me, logout, verify-email, resend-verification, profile, change-password, forgot/reset-password, record, history (GET/DELETE), history/export.csv, api-keys (GET/POST/DELETE), presets (GET/POST/GET id/DELETE). |
| mastering.py | — | Мастеринг: POST /api/master, POST /api/v2/master, POST /api/v2/batch, POST /api/v2/master/auto, GET /api/v2/chain/default, POST /api/v2/analyze, POST /api/v2/reference-match, POST /api/v2/upscale, POST /api/v2/isolate-vocal, статус/прогресс/результат/превью по job_id. |

---

## 2. Матрица: API → использование на фронте

### 2.1 Основное приложение (app.js + index.html)

| Метод | Эндпоинт | Где вызывается | Назначение |
|-------|----------|----------------|------------|
| GET | /api/version | app.js, index.html (футер), landing, pricing, status | Версия в футере |
| GET | /api/health | app.js (флаги функций, тир), login, register, status, landing (ссылка) | Флаги feature_*, vocal_isolation_enabled; редиректы при maintenance |
| GET | /api/debug-mode | app.js | Режим отладки |
| GET | /api/limits | app.js | Лимиты мастерингов (гость/пользователь) |
| GET | /api/ai/limits | app.js | Лимиты AI |
| GET | /api/presets/community | app.js | Пресеты сообщества в селекте |
| GET | /api/auth/presets | app.js | Пользовательские пресеты (при авторизации) |
| GET | /api/auth/presets/{id} | app.js | Загрузка одного пресета |
| POST | /api/auth/presets | app.js | Сохранение пресета |
| DELETE | /api/auth/presets/{id} | app.js | Удаление пресета |
| GET | /api/v2/chain/default | app.js | Цепочка по умолчанию (style, target_lufs) |
| POST | /api/v2/analyze | app.js | Анализ файла (LUFS, спектр и т.д.) |
| POST | /api/v2/master | app.js | Запуск мастеринга v2 |
| GET | /api/master/status/{job_id} | app.js, index.html (legacy) | Статус задачи |
| GET | /api/master/progress/{job_id} | app.js | SSE прогресс |
| GET | /api/master/result/{job_id} | app.js, index.html (legacy) | Скачивание результата |
| GET | /api/master/preview/{job_id} | app.js | Превью до/после (A/B) |
| POST | /api/v2/reference-match | app.js | Эталонное сведение |
| POST | /api/v2/master/auto | app.js | Авто-мастеринг (AI) |
| POST | /api/auth/record | app.js | Запись в историю после мастеринга |
| POST | /api/ai/recommend | app.js | Рекомендация пресета по файлу |
| POST | /api/ai/report | app.js | Отчёт по файлу |
| POST | /api/ai/nl-config | app.js | NL → настройки цепочки |
| POST | /api/ai/chat | app.js | Чат с AI |
| POST | /api/v2/batch | app.js | Пакетный мастеринг |
| POST | /api/v2/upscale | app.js | Апскейл (качество) |
| POST | /api/measure | index.html | Замер LUFS (простой поток) |
| POST | /api/master | index.html | Legacy: один мастеринг без v2 |
| GET | /api/master/status/{job_id} | index.html | Legacy |
| GET | /api/master/result/{job_id} | index.html | Legacy |

### 2.2 Страницы авторизации и профиля

| Метод | Эндпоинт | Страница | Назначение |
|-------|----------|----------|------------|
| POST | /api/auth/register | register.html | Регистрация |
| POST | /api/auth/login | login.html, admin.html | Вход |
| GET | /api/auth/me | admin.html | Проверка админа |
| GET | /api/auth/profile | profile.html | Профиль |
| GET | /api/auth/history | dashboard.html, profile.html | История мастерингов |
| DELETE | /api/auth/history/{id} | dashboard.html, profile.html | Удаление записи |
| GET | /api/auth/history/export.csv | profile.html | Экспорт истории |
| GET | /api/auth/api-keys | profile.html | Список API-ключей |
| POST | /api/auth/api-keys | profile.html | Создание ключа |
| DELETE | /api/auth/api-keys/{id} | profile.html | Отзыв ключа |
| POST | /api/auth/change-password | dashboard.html, profile.html | Смена пароля |
| POST | /api/auth/forgot-password | forgot-password.html | Запрос сброса |
| GET | /api/auth/verify-email | verify-email.html | Верификация по токену |
| POST | /api/auth/resend-verification | verify-email.html | Повтор письма |
| POST | /api/auth/reset-password | reset-password.html | Сброс по токену |

### 2.3 Тарифы и оплата

| Метод | Эндпоинт | Страница | Назначение |
|-------|----------|----------|------------|
| GET | /api/payments/plans | — | **Не используется:** список планов с бэка |
| POST | /api/payments/create | pricing.html | Создание платежа YooKassa |

На pricing.html ключи планов (`pro_month`, `studio_month` и т.д.) и подписи захардкожены; цены и список планов с сервера не подгружаются.

### 2.4 Админка (admin.html)

Все вызовы через `apiFetch(API + url)`. Используются:

- /api/auth/me, /api/auth/login
- /api/admin/stats
- /api/admin/users (GET, PATCH, DELETE), /api/admin/users/bulk-action, /api/admin/users/{id}/subscription
- /api/admin/transactions (GET, POST), /api/admin/transactions/export.csv
- /api/admin/news (GET, POST, PUT, DELETE)
- /api/admin/campaigns (GET, POST), /api/admin/campaigns/{id}/send
- /api/admin/settings (GET, PATCH)
- /api/admin/notifications/test-email, test-telegram
- /api/admin/llm/test
- /api/admin/prompts (GET, POST), /api/admin/prompts/{slug}/history, version/{id}, activate, reset
- /api/admin/audit
- /api/admin/reports/list, /api/admin/reports/{id}, summarize, export_raw.csv
- /api/admin/users/export.csv, /api/admin/backup/db

### 2.5 Остальные страницы

| Эндпоинт | Страница | Назначение |
|----------|----------|------------|
| GET /api/news | landing.html | Блок новостей (limit=5) |
| GET /api/version | landing.html, pricing.html | Футер |
| GET /api/health | status.html | Статус сервиса |

---

## 3. Эндпоинты без вызовов с фронта

| Эндпоинт | Описание | Статус |
|----------|----------|--------|
| **GET /api/presets** | Пресеты LUFS (spotify, youtube, club, broadcast и т.д.) | Не вызывается с фронта. Описан в docstring и в /docs; предназначен для API и POST /api/master (параметр preset). |
| **GET /api/payments/plans** | Список тарифов и цен с бэка | ✅ Подключён на pricing.html (очередь 10.1). |
| **POST /api/v2/isolate-vocal** | Отдельная изоляция вокала (Demucs), возврат WAV | ✅ В блоке «Изоляция вокала» добавлена кнопка «Только изолировать вокал» (очередь 10.2). |
| **GET /api/progress** | Тело PROGRESS.md (Markdown) | Для внешних клиентов и /progress.html; оставлен как есть. |
| **GET /api/styles** | Словарь стилей (имя → lufs) | Не вызывается с фронта. Описан в docstring; для API и /docs (полная цепочка — через GET /api/v2/chain/default). |
| **GET /api/extensions** | Статус расширений (community_presets_extra) | Не вызывается с фронта. Описан в docstring; для API и мониторинга. |

---

## 4. Дублирование и разночтения

### Потоки мастеринга (legacy vs v2)

- **Активный поток:** в приложении используется только **app.js**: при нажатии «Запустить мастеринг» вызывается **POST /api/v2/master**, затем ожидание по SSE (GET /api/master/progress/{job_id}) или polling (GET /api/master/status/{job_id}), затем GET /api/master/result/{job_id}. Параметры цепочки, PRO-модули и стили передаются через форму.
- **Legacy-поток (не активен):** в index.html в **закомментированном** блоке (fallback при отсутствии app.js) описан старый поток: POST /api/measure, **POST /api/master** (без v2), опрос /api/master/status, GET /api/master/result. Эндпоинт POST /api/master сохранён на бэкенде для обратной совместимости и скриптов; UI его не вызывает при нормальной загрузке app.js.

- **Проверка авторизации:** в разных местах используется `localStorage.getItem('magic_token')` и при необходимости заголовок `Authorization: Bearer ...`. Единого слоя auth (например, перехват 401 и редирект на /login) может не хватать — при необходимости вынести в общий хелпер.

- **Базовый URL API:** в app.js используется переменная `API` (корень бэка). В части страниц вызовы идут как `fetch('/api/...')` без префикса — допустимо при раздаче с того же origin; при отдельном фронте нужна единая конфигурация (например, через `API` везде).

---

## 5. Связь с другими документами

- **План доработок:** [PLAN_DORABOTKI.md](PLAN_DORABOTKI.md) — задачи по приоритету.
- **Функции и уровни:** [ФУНКЦИИ_И_УРОВНИ.md](ФУНКЦИИ_И_УРОВНИ.md) — что доступно Free/Pro/Studio.
- **Версионность:** [VERSIONING.md](VERSIONING.md) — правила версий и CHANGELOG.

---

## 6. Краткие выводы

1. **Покрытие:** большинство эндпоинтов используются. GET /api/presets, GET /api/styles, GET /api/extensions не вызываются с фронта и задокументированы для API (/docs).
2. **Тарифы:** GET /api/payments/plans подключён на pricing.html (очередь 10.1).
3. **Изоляция вокала:** кнопка «Только изолировать вокал» добавлена (очередь 10.2); эндпоинт также используется в пайплайне мастеринга.
4. **Потоки мастеринга:** активный UI — только v2 (app.js, POST /api/v2/master); код с POST /api/master в index.html закомментирован (fallback). POST /api/master на бэке сохранён для совместимости.
5. **Админка и авторизация:** связка admin.html ↔ /api/admin и страницы входа/профиля ↔ /api/auth согласованы.

Дальнейшие шаги — см. раздел «План доработок по аудиту» в [PLAN_DORABOTKI.md](PLAN_DORABOTKI.md).
