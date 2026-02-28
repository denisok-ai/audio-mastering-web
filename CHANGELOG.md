# Changelog — Magic Master

All notable changes to this project are documented here.  
Format: `[Phase] Brief description — files changed`.

---

## [Unreleased]

---

## [0.4.0] — 2026-02-28

### Крупные изменения продукта (версия увеличена: 0.3.0 → 0.4.0)

- **Редизайн админ-панели:** настройки из .env в БД и UI (табы Общие, SMTP, YooKassa, Telegram, LLM), PATCH /api/admin/settings, промпты с версиями и историей, отчётность (10 отчётов, резюме LLM, экспорт CSV), журнал действий с фильтрами.
- **Режим обслуживания и флаги функций:** maintenance_mode (503 для не-админов), feature_ai_enabled / feature_batch_enabled / feature_registration_enabled (проверки в API и скрытие блоков на фронте).
- **Защита от LLM-инъекций:** модуль llm_guard.py, настраиваемая проверка ввода (запрещённые подстроки, regex, лимиты длины), настройки во вкладке LLM админки.
- **Пагинация во всех закладках админки:** Транзакции, Новости, Рассылки (limit/offset, total), единообразно 20 записей на страницу.
- **Тестовые данные:** скрипт backend/scripts/seed_admin_data.py — 24 пользователя, 44 транзакции, 15 новостей, 12 рассылок, 44 записи журнала аудита.
- **Версионность:** версия и дата сборки в backend/app/version.py, экспорт в /api/version и /api/health; правило обновления версии при крупных изменениях зафиксировано в комментарии в version.py.

---

### P64 — Пресеты сообщества
- **backend/app/presets_community.json**: файл с пресетами (id, name, target_lufs, style, chain_config). Три примера: Stream (−14 LUFS), Подкаст (−16), Клуб (−9).
- **backend/app/main.py**: _load_community_presets(), GET /api/presets/community — возвращает список из JSON.
- **frontend/app.js**: в селекте пресетов — optgroup «Пресеты сообщества» (значения c:id); загрузка без логина; applyPresetToUI(); кнопка «Удалить» отключена для пресетов сообщества.
- **backend/tests/test_api.py**: test_api_presets_community.

### P63 — E2E-тест полного цикла мастеринга
- **backend/tests/test_e2e_mastering.py**: тест test_e2e_mastering_flow — POST /api/v2/master с WAV-файлом (фикстура minimal_wav_bytes), опрос GET /api/master/status/{job_id} до status=done, GET /api/master/result/{job_id}, проверка, что ответ — WAV (RIFF, WAVE).

### P62 — Вынос хелперов из main.py (рефакторинг)
- **backend/app/helpers.py**: новый модуль с функциями get_client_ip(request), allowed_file(filename), check_audio_magic_bytes(data, filename). Зависит только от app.config (settings).
- **backend/app/main.py**: импорт этих функций из .helpers с алиасами (_get_client_ip, _allowed_file, _check_audio_magic_bytes); удалены локальные определения. Объём main.py уменьшен.

### P61 — Доступность (a11y)
- **frontend/index.html**: aria-label на кнопках без текста (сброс файла, play/pause, убрать эталон, отправить в чат); role="status", aria-live="polite", aria-label на блок прогресса мастеринга; role="region", aria-live="polite" на контейнер тостов. CSS: видимый фокус (outline) для button:focus-visible, a:focus-visible.

### P60 — Проверка magic bytes при загрузке аудио
- **backend/app/main.py**: функция `_check_audio_magic_bytes(data, filename)` — проверка сигнатур WAV (RIFF…WAVE), FLAC (fLaC), MP3 (ID3 или 0xFF 0xE?). Вызов после чтения файла во всех эндпоинтах загрузки (measure, master, v2/master, v2/batch, v2/master/auto, v2/analyze, v2/reference-match, AI recommend/report). При несоответствии — HTTP 400.
- **backend/tests/test_api.py**: тест `test_check_audio_magic_bytes` для валидных и невалидных сигнатур.

### P59 — Базовая i18n (локализация)
- **backend/app/main.py**: добавлен `GET /api/locale` — доступные локали `["ru","en"]`, значение по умолчанию `ru`.
- **frontend/locales/ru.json, en.json**: словари строк для интерфейса (app.title, app.measure, app.upload, app.download_report, locale.label и др.).
- **frontend/index.html**: переключатель языка RU/EN в шапке; атрибуты `data-i18n` на заголовок, кнопки «Измерить громкость», «Скачать отчёт», карточку «Загрузка файла».
- **frontend/app.js**: логика i18n — getLocale/setLocale (localStorage и ?lang=), loadLocale (fetch /locales/{lang}.json), applyI18n() для элементов с data-i18n, привязка к кнопкам переключения.
- **backend/tests/test_api.py**: тест `test_api_locale` для эндпоинта /api/locale.

### P58 — Метрики для мониторинга
- **backend/app/main.py**: добавлен `GET /api/metrics` — плоский JSON: `uptime_seconds`, `jobs_running`, `jobs_total`, `version` для внешнего скрапинга и дашбордов.
- **backend/tests/test_api.py**: тест `test_api_metrics` проверяет ответ эндпоинта.
- **doc/RUNBOOK.md**: в разделе «Проверка состояния» добавлено упоминание `/api/metrics`.

### P57 — Экспорт AAC (M4A)
- **backend/app/pipeline.py**: экспорт в формат `aac` (контейнер M4A, 192 kbps) через pydub/ffmpeg (`format="ipod"`, `codec="aac"`).
- **backend/app/main.py**: формат `aac` разрешён во всех эндпоинтах мастеринга и ref-match; для имени файла используется расширение `.m4a`; в `_mime_map` для preview добавлены `aac` и `m4a` → `audio/mp4`.
- **frontend/index.html**: в селект формата экспорта добавлена опция «AAC — 192 kbps (M4A) 🔒» (tier pro).
- **frontend/app.js**: при скачивании результата для формата aac используется расширение `.m4a`; текст модалки апгрейда обновлён (MP3, FLAC, OPUS, AAC).
- **backend/tests/test_pipeline.py**: тест `test_export_audio_aac_returns_bytes` (пропуск при отсутствии ffmpeg).

### План эксплуатации: пауза разработки, runbook, бэкап, логирование
- **PROGRESS.md**: Режим «пауза разработки новых фич»; текущая версия готова к эксплуатации; пункты P57+ переведены в бэклог.
- **doc/RUNBOOK.md**: Runbook для оператора — проверка health, логи, бэкап БД, перезапуск сервиса, напоминание про CORS и webhook в production.
- **DEPLOY.md**: Секция «Production: безопасность (CORS и webhook)»; секция «Автоматический бэкап БД» с описанием cron и скрипта; в шпаргалку добавлена проверка /api/health и ссылка на RUNBOOK.
- **deploy/backup_db.sh**: Скрипт автобэкапа SQLite (VACUUM INTO) с настраиваемыми BACKEND_DIR и BACKUP_DIR.
- **backend/app/main.py**: Логирование ошибок (logger.error) при сбое загрузки аудио (v2/master, ai/recommend) и при падении фоновой задачи мастеринга (job_id, filename, краткое описание).
- **backend/app/payments.py**: Логирование ошибки при некорректном JSON в webhook YooKassa.

### P56 — CORS и безопасность webhook YooKassa
- **backend/app/config.py**: `cors_origins` (MAGIC_MASTER_CORS_ORIGINS), `yookassa_webhook_ip_whitelist` (MAGIC_MASTER_YOOKASSA_WEBHOOK_IP_WHITELIST).
- **backend/app/main.py**: CORS берёт список origins из настроек; пусто — разрешены все (*).
- **backend/app/payments.py**: при заданном whitelist webhook принимается только с указанных IP; исправлена передача amount_val и currency_val в notify_payment.
- **.env.example**: добавлены MAGIC_MASTER_CORS_ORIGINS и MAGIC_MASTER_YOOKASSA_WEBHOOK_IP_WHITELIST с комментариями.

### P55 — Spectral Denoiser: пресеты и опциональный порог
- **backend/app/pipeline.py**: `DENOISE_PRESETS` (light 0.25/20%, medium 0.5/15%, aggressive 0.75/10%); `apply_spectral_denoise` уже принимал `noise_percentile`.
- **backend/app/main.py**: в `POST /api/v2/master` добавлены `denoise_preset` (light|medium|aggressive), `denoise_noise_percentile` (5–40); в `_run_mastering_job_v2` при наличии пресета используются значения из `DENOISE_PRESETS`.
- **frontend/index.html**: в карточке Spectral Denoiser — селект «Пресет»: Свой / Лёгкий / Средний / Агрессивный; стили `.pro-select-row`, `.pro-select`.
- **frontend/app.js**: в `collectProModuleParams` при включённом Denoiser отправляется `denoise_preset` при выборе пресета, иначе `denoise_strength`.
- **backend/tests/test_pipeline.py**: тест `test_denoise_presets` для `DENOISE_PRESETS` и `apply_spectral_denoise`.

### P54 — pytest pipeline и AI
- **backend/tests/test_pipeline.py**: тесты remove_dc_offset, measure_lufs, compute_spectrum_bars, compute_vectorscope_points, compute_lufs_timeline, measure_stereo_correlation, export_audio (wav), run_mastering_pipeline, load_audio_from_bytes, STYLE_CONFIGS.
- **backend/tests/test_ai.py**: тесты get_ai_limit_for_tier, check_ai_rate_limit, record_ai_usage, recommend_preset (rule-based), report_with_recommendations, nl_to_config (без API), VALID_STYLES.

### P53 — Admin: Test SMTP & Telegram
- **backend/app/admin.py**: `POST /api/admin/notifications/test-email` (отправляет тестовое письмо), `POST /api/admin/notifications/test-telegram`.
- **frontend/admin.html**: кнопки «📧 Тест Email» и «✈ Тест Telegram» с результатом на странице Настройки.

### P52 — API Keys for Pro/Studio
- **backend/app/database.py**: модель `ApiKey` (id, user_id, name, key_prefix, key_hash SHA-256, is_active, last_used_at); функции `create_api_key`, `get_api_keys_for_user`, `revoke_api_key`, `get_user_by_api_key`.
- **backend/app/main.py**: эндпоинты `GET/POST /api/auth/api-keys`, `DELETE /api/auth/api-keys/{id}`; поддержка `X-API-Key` заголовка в `_get_current_user_optional`; лимит 10 ключей.
- **frontend/profile.html**: раздел «API-ключи» (только для Pro/Studio) — список, создание (ключ показывается однократно), кнопка копирования, отзыв.

### P51 — Telegram Admin Notifications
- **backend/app/notifier.py**: новый модуль; `notify()` (async в фоне через threading); шаблоны `notify_new_user`, `notify_payment`, `notify_payment_failed`, `notify_mastering_error`, `notify_server_startup`, `notify_backup_done`, `notify_user_blocked`.
- **backend/app/config.py**: поля `telegram_bot_token`, `telegram_admin_chat_id`.
- **backend/app/main.py**: вызовы `notify_new_user` при регистрации, `notify_mastering_error` при ошибке задачи, `notify_server_startup` при старте.
- **backend/app/payments.py**: вызов `notify_payment` при успешном webhooks.
- **.env.example**: добавлены переменные `MAGIC_MASTER_TELEGRAM_BOT_TOKEN/CHAT_ID` с инструкцией.

### P50 — Admin Database Backup
- **backend/app/admin.py**: `GET /api/admin/backup/db` — горячий бэкап SQLite через `VACUUM INTO` во временный файл → скачивание; имя файла содержит метку времени.
- **backend/app/database.py**: экспортирован `DATABASE_URL` для определения пути к файлу БД.
- **frontend/admin.html**: кнопка «⬇ Backup DB» на странице Настройки.

### P49 — PWA (Progressive Web App)
- **frontend/manifest.json**: Web App Manifest — имя, тема, иконки 192/512, shortcuts (Мастеринг, Профиль).
- **frontend/sw.js**: Service Worker — Cache-First для статики (JS/CSS/шрифты), Network-First для API, Stale-While-Revalidate для HTML, исключение SSE/preview.
- **backend/app/main.py**: маршруты `/sw.js` (с `Service-Worker-Allowed: /`) и `/manifest.json`.
- **frontend/index.html**: `<link rel="manifest">`, `<meta name="theme-color">`, Apple PWA мета-теги.
- **frontend/app.js**: регистрация SW + обработчик `beforeinstallprompt` с кнопкой «⬇ Установить».

### P48 — Admin Bulk Actions
- **backend/app/admin.py**: `POST /api/admin/users/bulk-action` — действия `block/unblock/delete/set_tier` для списка user_ids; защита от самоудаления; подробный ответ (affected/skipped).
- **frontend/admin.html**: чекбоксы в каждой строке таблицы пользователей; «Select All»; плавающий `bulk-bar` с кнопками блок/разблок/тариф/удалить + снять выбор; бейдж `unverified`.

### P47 — Service Status Page
- **backend/app/main.py**: `GET /api/health` расширен — компоненты (БД, диск, ffmpeg), активные задачи, версия, uptime, Python. Маршрут `/status`.
- **frontend/status.html**: публичная страница с баннером OK/degraded/error, карточками компонентов, статистикой; автообновление каждые 30 с.

### P46 — Global API Rate Limit
- **backend/app/main.py**: middleware `global_rate_limit_middleware` — 300 req/min с IP для всех `/api/*` эндпоинтов; исключение SSE (`/api/master/progress/`); заголовок `Retry-After`.
- **backend/app/config.py**: поле `global_rate_limit` (по умолчанию 300); `MAGIC_MASTER_GLOBAL_RATE_LIMIT`.
- **.env.example**: добавлена переменная.

### P45 — In-Browser A/B Audio Player
- **backend/app/main.py**: `GET /api/master/preview/{job_id}?src=original|mastered` — стриминг аудио без скачивания; оригинал сохраняется в `_jobs[job_id]`.
- **frontend/index.html**: блок плеера `.ab-player-wrap` с HTML5 `<audio>`, прогресс-баром, громкостью, A/B кнопками.
- **frontend/app.js**: `window.initABPlayer(jobId)`, управление play/pause/seek, переключение A/B с сохранением позиции, сброс по событию `masteringReset`.

### P44 — CHANGELOG
- Added this `CHANGELOG.md`.

### P43 — Admin Dashboard Analytics
- **backend/app/admin.py**: `GET /api/admin/stats` расширен — выручка всего, активные подписки, неподтверждённые, массивы `by_day` (7 дней) для пользователей / мастерингов / выручки.
- **frontend/admin.html**: SVG-спарклайны (`drawSparkline`) для новых пользователей, мастерингов и выручки; прогресс-бар распределения тарифов (`drawTierBar`); новые карточки (активные подписки, выручка всего).

### P42 — User History CSV Export
- **backend/app/main.py**: `GET /api/auth/history/export.csv` — скачать историю мастерингов в CSV (UTF-8 BOM).
- **frontend/profile.html**: кнопка «⬇ CSV» в заголовке блока истории.

### P41 — Email Verification
- **backend/app/config.py**: поле `require_email_verify` (по умолчанию `False`); переменная `MAGIC_MASTER_REQUIRE_EMAIL_VERIFY`.
- **backend/app/database.py**: поле `User.is_verified` (Boolean, default `True` для legacy); миграция колонки `is_verified`.
- **backend/app/mailer.py**: функция `send_email_verification(to, verify_url)`.
- **backend/app/main.py**: хранилище `_verify_tokens`; эндпоинты `GET /api/auth/verify-email`, `POST /api/auth/resend-verification`; логика в `api_auth_register` (при `require_email_verify=True` создаёт аккаунт с `is_verified=False`); проверка `is_verified` в `api_auth_login`; страница `/verify-email`.
- **frontend/verify-email.html**: страница подтверждения с тремя состояниями (ожидание, успех, ошибка + форма повторной отправки).
- **.env.example**: добавлена переменная `MAGIC_MASTER_REQUIRE_EMAIL_VERIFY`.

---

## v0.10 — Production Docker + Nginx (P40)
- **docker-compose.yml**: Compose для локальной разработки (порт 8000, volume для SQLite).
- **docker-compose.prod.yml**: Compose для production — сервисы `app` + `nginx` с healthcheck.
- **deploy/nginx/magic-master.conf**: полная production-конфигурация Nginx — HTTPS redirect, TLS 1.2/1.3, security headers, SSE-прокси (`proxy_buffering off`, `X-Accel-Buffering no`), large uploads (210 MB).
- **.env.example**: все переменные окружения с комментариями.

## v0.9 — Admin CSV Export (P39)
- **backend/app/admin.py**: `GET /api/admin/users/export.csv`, `GET /api/admin/transactions/export.csv` (UTF-8 BOM).
- **frontend/admin.html**: кнопки «⬇ CSV» в секциях Users и Transactions.

## v0.9 — pytest Auth (P38)
- **backend/tests/test_auth.py**: 26 тестов — register, login, rate limit (P33), profile (P31), change-password (P34), forgot/reset password (P35), history, logout.

## v0.8 — SSE Progress (P37)
- **backend/app/main.py**: `GET /api/master/progress/{job_id}` — Server-Sent Events для прогресса мастеринга.
- **frontend/app.js**: `waitForJobCompletion()` с SSE + fallback polling; `_pollJobCompletion()`.
- **deploy/nginx/magic-master.conf**: блок `location ~ ^/api/master/progress/` с `proxy_buffering off`.

## v0.8 — DB Migrations (P36)
- **backend/app/database.py**: `_run_migrations()` — безопасное добавление новых колонок через `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` без потери данных.

## v0.7 — Password Reset (P35)
- **backend/app/main.py**: `POST /api/auth/forgot-password`, `POST /api/auth/reset-password` (in-memory токены с TTL 1 ч).
- **backend/app/mailer.py**: `send_password_reset_email(to, reset_url)`.
- **frontend/forgot-password.html**, **frontend/reset-password.html**: новые страницы.
- **frontend/login.html**: ссылка «Забыли пароль?».

## v0.7 — Auth Rate Limit (P33)
- **backend/app/main.py**: `_check_auth_rate_limit(ip)` — 10 попыток/мин; применено к `/api/auth/login` и `/api/auth/register`.

## v0.7 — Subscription Email Warnings (P32)
- **backend/app/mailer.py**: `send_subscription_expiry_warning_email`, `send_subscription_expired_email`.
- **backend/app/database.py**: поле `subscription_warning_sent`; логика в `check_and_expire_subscription`.

## v0.7 — User Profile Page (P31)
- **backend/app/main.py**: `GET /api/auth/profile`; маршрут `/profile`.
- **frontend/profile.html**: тариф, статус подписки, история мастерингов, смена пароля.

## v0.6 — Admin Settings (P29)
- **backend/app/admin.py**: `GET /api/admin/settings` — замаскированные SMTP / YooKassa / App настройки.
- **frontend/admin.html**: вкладка «Настройки» в сайдбаре.

## v0.6 — Subscription Expiry Check (P28)
- **backend/app/database.py**: `check_and_expire_subscription(db, user_id)` — авто-даунгрейд + email.
- **backend/app/main.py**: вызов в `_get_current_user_optional`.

## v0.5 — OPUS Export (P30)
- **backend/app/pipeline.py**: экспорт в OPUS 192 kbps через pydub/libopus.
- **frontend/index.html**: опция «OPUS — 192 kbps 🔒» в селекте форматов.

## v0.5 — PDF Report Export (P27)
- **frontend/app.js**: `buildReportHtmlForPrint(data)` + кнопка «PDF» через `window.print()`.
- **frontend/index.html**: кнопка «PDF» в панели отчётов.

## v0.5 — pytest Admin & Payments (P26)
- **backend/tests/test_api.py**: 17 тестов — CRUD users, news, campaigns, transactions, YooKassa webhook.

## v0.4 — YooKassa Payments (P23)
- **backend/app/payments.py**: `POST /api/payments/create`, `POST /api/payments/webhook`.
- **frontend/pricing.html**: страница тарифов с кнопками оплаты.

## v0.4 — Email Marketing (P22)
- **backend/app/mailer.py**: SMTP + `send_welcome_email`, `send_campaign_email`.
- **backend/app/admin.py**: эндпоинты кампаний `POST /api/admin/campaigns/{id}/send`.

## v0.3 — Admin Panel (P18–P21)
- **backend/app/admin.py**: CRUD для users, news, campaigns, transactions; JWT-защита admins.
- **frontend/admin.html**: SPA-панель администратора с сайдбаром и всеми разделами.

## v0.3 — AI Agents (P13–P17)
- **backend/app/ai_agents.py**: preset recommendation, report interpretation, auto-mastering, NL→config, chat assistant.
- **frontend/index.html**, **frontend/app.js**: AI-кнопки, AI-чат, NL-режим.

## v0.2 — Audio Analysis & Visualizers (P6–P12)
- **backend/app/pipeline.py**: анализ LUFS/peak/dynamics; цепочка обработки; экспорт WAV/MP3/FLAC.
- **frontend/app.js**: визуализаторы waveform, spectrum, vectorscope, LUFS-timeline.
- **frontend/index.html**: полный UI обработки.

## v0.1 — Auth & Core (P0–P5)
- **backend/app/database.py**: SQLAlchemy models (User, MasteringRecord, Transaction, NewsPost, EmailCampaign, UserPreset).
- **backend/app/auth.py**: JWT, bcrypt, create_user, get_user_by_email.
- **backend/app/main.py**: FastAPI, `/api/auth/register`, `/api/auth/login`, `/api/auth/me`.
- **frontend/index.html**, **frontend/login.html**, **frontend/register.html**: базовый UI.
- **backend/requirements.txt**: все зависимости.
- **start.sh**: скрипт запуска для разработки.

---

*Changelog автоматически не генерируется — обновляйте вручную при каждом релизе.*
