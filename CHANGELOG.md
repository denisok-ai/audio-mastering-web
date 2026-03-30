# Changelog — Magic Master

All notable changes to this project are documented here.  
Format: `[Phase] Brief description — files changed`.

---

## [Unreleased]

---

## [0.7.1] — 2026-03-29

### Исправления и эксплуатация
- **Авторизация:** проверка JWT перед редиректом на `/login` и `/register`; очистка «зомби»-токена при 401 в **frontend/login.html**, **register.html**, **frontend/app.js**.
- **Лендинг:** нейтральный CTA и блок статистики без вводящего в заблуждение «Войти»/«0» — **frontend/locales/site-ru.json**, **site-en.json**, **frontend/landing.html**.
- **DSP:** сглаживание старта трека и границы lookahead — **backend/app/pipeline.py**; тот же output fade для v2 — **backend/app/routers/mastering.py**; тесты в **backend/tests/test_pipeline.py**.
- **Отладка мастеринга:** флаги `MAGIC_MASTER_MASTERING_TRACE` / `MAGIC_MASTER_MASTERING_TRACE_LUFS_STAGES`, модуль **backend/app/mastering_trace.py**, трассировка этапов в pipeline/chain/роутере/боте.
- **Деплой:** рекомендации по ротации логов (journald) — **deploy/journald/README.md**, пример **deploy/journald/99-magic-master-journal.conf**.

---

## [0.7.0] — 2026-03-29

### Фаза 1: SEO-лендинги, блог, LUFS-инструмент, рефералы, брендинг экспорта
- **frontend/suno-mastering.html**, **udio-mastering.html**, **podcast-mastering.html**, **telegram-bot.html**, **lufs-analyzer.html**, **referral.html**: публичные SEO-страницы (GA, Метрика, Clarity), демо-плеер Suno (`/demos/*.wav`), ссылки в **frontend/landing.html** (навигация, таблица сравнения с LANDR/eMastered), **frontend/pricing.html**, **frontend/sitemap.xml**.
- **doc/SEO_PAGE_SNIPPET.md**: эталон для новых страниц.
- **content/blog/**: статьи `mastering-suno-ai.md`, `chto-takoe-lufs.md`, `mastering-podcasta.md`.
- **backend/app/routers/blog.py**: `/blog`, `/blog/{slug}` — Markdown + YAML frontmatter из `content/blog/` (fallback без `markdown`/`pyyaml`).
- **backend/app/routers/tools.py**: `POST /api/tools/lufs-analyze` (до 50 МБ, лимит по IP/час из `MAGIC_MASTER_LUFS_TOOL_RATE_PER_HOUR`).
- **backend/app/routers/referral.py**: `GET /api/referral/my-link`, `/api/referral/stats`; модель **Referral**, поля **users.referral_code**, **referred_by_user_id**; хук регистрации (**routers/auth.py**, `?ref=`) и награда пригласившему после первого успешного мастеринга (**mastering.py**); настройки **MAGIC_MASTER_REFERRAL_REWARD_INVITER / INVITEE**.
- **backend/app/metadata.py**: теги MASTERED_BY / ENCODER в MP3 и FLAC после экспорта (**mastering.py**).
- **backend/app/services/share_card.py**, **GET /api/master/share/{job_id}**: PNG 1200×630 (Pillow); **frontend/app.js** — автоскачивание карточки до выдачи результата.
- **backend/requirements.txt**: markdown, PyYAML, mutagen, Pillow.
- **backend/app/main.py**: маршруты новых страниц; подключены роутеры blog, tools, referral.

---

## [0.6.8] — 2026-03-28

### P0 pre-launch: цены/лимиты, SEO, аналитика, конверсия
- **frontend/landing.html**, **frontend/pricing.html**, **frontend/locales/ru.json**, **frontend/locales/en.json**: канон цен Pro/Studio (1000/2500 ₽, год 10k/25k), лимиты гостя/Pro/Studio, 8 пресетов, размеры файлов 100/300/800 МБ, FAQ без устаревших «3/день» и «в разработке»; блок новостей с лендинга убран (SEO).
- **frontend/app.js**: дефолт `_tierInfo` 1/1, тексты лимитов без «3 в день»; после успешного мастера гостю — модалка с призывом к регистрации (раз на сессию, `sessionStorage`).
- **frontend/index.html**: og:image, Clarity, актуальные формулировки в upgrade-модалке, ссылка «Зарегистрироваться».
- **frontend/login.html**, **register.html**: meta description, canonical, `noindex`, Яндекс.Метрика, Clarity.
- **frontend/404.html**, **429.html**, **500.html**: `robots noindex,nofollow`.
- **backend/app/config.py**: `clarity_project_id` (`MAGIC_MASTER_CLARITY_PROJECT_ID`).
- **backend/app/main.py**: `/og-image.png` (1200×630 PNG-заглушка), `/analytics/clarity.js` (подгрузка Clarity при заданном ID).
- **.env.example**: комментарий к Clarity.

---

## [0.6.7] — 2026-03-28

### Бот уведомлений — ответы при Topics в личке
- **backend/app/bot/notify_handlers.py**: `_safe_answer` — при `TelegramBadRequest` повтор через `send_message` без топика (исправляет «message thread not found» и молчание кнопок / отсутствие клавиатуры на мобильном).
- **backend/app/bot/notify_webhook_route.py**: при ошибке обработки апдейта всё равно **200** и `{"ok": true}`, чтобы Telegram не крутил бесконечные ретраи.

---

## [0.6.6] — 2026-03-28

### Бот уведомлений — админское меню без боковой панели команд
- **backend/app/bot/keyboards.py**: `admin_menu_button_rows`, `admin_menu_reply_markup_dict`, `all_admin_menu_button_texts`, `admin_menu_reply` (RU/EN).
- **backend/app/bot/notify_handlers.py**: нижнее меню — Сервер, Статистика, Задачи, Ошибки, Здоровье, Пользователи, Выручка, Отчёт, Рассылка, Помощь; отчёты как в user bot `/admin`; `/broadcast`; Помощь — ссылка на клиентский бот.
- **backend/app/notifier.py**: в уведомлениях — админская клавиатура.
- **backend/app/bot/lifecycle.py**: `notify_bot_startup` — `delete_my_commands` (убрать меню «/» слева).
- **backend/tests/test_bot_lifecycle.py**: тест разметки админ-меню.

---

## [0.6.5] — 2026-03-28

### Бот уведомлений (magicmaster.pro) — меню как у клиентского бота
- **backend/app/notifier.py**: к каждому уведомлению в `TELEGRAM_ADMIN_CHAT_ID` добавляется та же **ReplyKeyboard**, что у user bot (Мастеринг, Анализ, Пресеты, AI чат, Баланс, Помощь).
- **backend/app/bot/keyboards.py**: `main_menu_button_rows`, `main_menu_reply_markup_dict`, `all_main_menu_button_texts` — единый источник подписей.
- **backend/app/bot/notify_bot_setup.py**, **notify_handlers.py**, **notify_webhook_route.py**: webhook `POST /bot/notify/webhook` для бота уведомлений; `/start`, кнопки меню — подсказка и ссылка на клиентский бот (`MAGIC_MASTER_USER_BOT_TELEGRAM_URL`).
- **backend/app/bot/lifecycle.py**: `notify_bot_startup` / `notify_bot_shutdown` (токен не должен совпадать с `USER_BOT_TOKEN`).
- **backend/app/config.py**: `telegram_bot_webhook_secret`, `user_bot_telegram_url`.
- **.env.example**: комментарии к новым переменным.

---

## [0.6.4] — 2026-03-28

### Telegram user bot — раздельное меню «/»
- **backend/app/bot/lifecycle.py**: `BotCommandScopeDefault` — 12 команд без `/admin` и без служебных админ-команд; для каждого `telegram_id` с `is_admin` + привязкой — `BotCommandScopeChat` с расширенным меню (+ `/admin`, `/server`, `/stats`, `/jobs`, `/errors`, `/report`, `/broadcast`). `refresh_menu_for_telegram_chat` после успешного `/code` и `/unlink`.
- **backend/app/database.py**: `list_admin_telegram_ids(db)`.
- **backend/app/bot/handlers/account.py**: обновление меню после привязки/отвязки.
- **backend/tests/test_bot_lifecycle.py**: проверки списков команд.

---

## [0.6.3] — 2026-03-28

### Telegram AI-консультант
- **backend/app/bot/handlers/ai_chat.py**: вызов `chat_assistant` через `asyncio.to_thread` — не блокирует event loop во время запроса к DeepSeek/OpenAI; сессия БД закрывается до LLM; при ошибке — `logger.exception` и ответ пользователю `txt(lang, "error")`.

---

## [0.6.2] — 2026-03-28

### Telegram user bot — меню команд и webhook
- **backend/app/main.py**: `logging.getLogger("app.bot").setLevel(INFO)` в lifespan — сообщения lifecycle видны в journald.
- **backend/app/bot/lifecycle.py**: при старте `set_my_commands` — список команд в меню «/» (RU); логирование успеха/ошибки; при сбое `set_webhook` — `logger.exception` (видно в journalctl), лог «secret=yes|no»; `StreamHandler` для `app.bot` → stderr/journald.
- **backend/tests/test_bot_lifecycle.py**: проверка списка команд.

---

## [0.6.1] — 2026-03-28

### Админ-панель в Telegram-боте (RU)
- **backend/app/bot/server_metrics.py**: метрики сервера (psutil + fallback), отчёты на русском.
- **backend/app/bot/admin_reports.py**: форматирование статистики, задач, здоровья, выручки, ошибок, полного отчёта.
- **backend/app/bot/handlers/admin.py**: меню `/admin` (8 кнопок), команды `/server`, `/jobs`, `/errors`, `/report`; длинные отчёты режутся по лимиту Telegram.
- **backend/app/bot/keyboards.py**: русские подписи кнопок админки.
- **backend/app/bot/anomaly_monitor.py**: фоновые проверки при `alert_monitoring_enabled` — CPU, RAM, диск, RSS процесса, доля ошибок мастеринга за час, порог очереди.
- **backend/app/notifier.py**: `notify_operational_anomaly` для алертов на русском.
- **backend/app/jobs_store.py**: `list_recent_error_jobs()` для админки.
- **backend/app/config.py**: пороги `anomaly_*`.
- **backend/requirements.txt**: `psutil`.
- **backend/tests/test_admin_bot_metrics.py**, **doc/TELEGRAM_USER_BOT.md**, **.env.example**: документация и примеры.

---

## [0.6.0] — 2026-03-28

### AI Consultant Bot (DeepSeek)
- **backend/app/bot/knowledge_base.py**: двуязычная база знаний (RU + EN) — стили, тарифы, FAQ, PRO-модули, команды бота, рекомендации по AI-музыке, конверсионные фразы. Функция `get_product_knowledge(lang)`.
- **backend/app/ai.py**: обновлён системный промпт консультанта — `PRODUCT_KNOWLEDGE` из knowledge_base, ответ на языке пользователя. Контекст увеличен с 800 до 4000 символов, max_tokens с 500 до 1000. Поле `hint` из контекста больше не дублируется в промпте.
- **backend/app/bot/handlers/ai_chat.py**: исправлен `NameError: settings` (добавлен `from ...config import settings`). Удалён устаревший `BOT_CHAT_CONTEXT` — знания теперь из knowledge_base. Передаётся `lang` из Telegram-профиля.
- **frontend/app.js**: веб-чат передаёт `lang` (из UI-локали) в контекст `/api/ai/chat` для выбора нужной KB.
- **.env.example**, **deploy/env.production**: `AI_BACKEND=deepseek` по умолчанию, документация DeepSeek.
- **backend/tests/test_bot_knowledge.py**: 7 тестов на KB и интеграцию с `chat_assistant`.

### Telegram user bot
- **backend/app/bot/**: клиентский бот (aiogram 3) — `/start`, мастеринг, анализ, AI `/ask`, привязка `/link` + `/code`, админ `/admin`, webhook `/bot/webhook`, push при готовности мастеринга с сайта.
- **backend/app/database.py**: `telegram_id`, `telegram_lang`, таблицы `telegram_link_codes`, `telegram_engagement`.
- **deploy/nginx/magic-master-proxy.inc**: `location /bot/`.
- **doc/TELEGRAM_USER_BOT.md**: инструкция по настройке.

---

## [0.5.7] — 2026-03-28

### Безопасность и оптимизация
- **backend/app/auth.py:** предупреждение CRITICAL при использовании JWT-секрета по умолчанию.
- **backend/app/main.py:** скрытие `/docs`, `/redoc`, `/openapi.json` в production (debug=0); CORS: запрет `credentials=True` с wildcard origins; `/api` root не показывает ссылку на docs в production.
- **backend/app/routers/mastering.py:** санитизация ошибок — внутренние детали исключений не возвращаются клиенту в HTTP 500.
- **backend/run_production.py:** `limit_max_requests=10000`, `h11_max_incomplete_event_size=16384` для защиты от утечек памяти.
- **deploy/systemd/magic-master.service:** `MemoryHigh=9G` (soft limit), `OnFailure=magic-master-alert@%n.service`.
- **deploy/systemd/magic-master-alert@.service:** шаблон для email-алерта при падении сервиса.

### Серверная инфраструктура
- **deploy/backup_full.sh:** полный бэкап (SQLite + .env + nginx), gzip, ротация с гарантией 15% свободного места.
- **deploy/disk_monitor.sh:** мониторинг диска, автоочистка tmp/журналов, email-алерт при дефиците.
- **deploy/send_alert.sh:** отправка email-алертов через msmtp.
- **deploy/logrotate/magic-master:** ротация логов nginx и бэкапов (14 дней, compress).
- **deploy/journald.conf.d/magic-master.conf:** ограничение журнала systemd (500M, 14 дней).
- **deploy/nginx/gzip.conf:** расширенная gzip-конфигурация (JS, JSON, SVG, шрифты и др.).
- **deploy/nginx/magic-master-proxy.inc:** кеш статики 7 дней, `no-cache` для HTML.
- **deploy/env.production:** оптимизированный .env для сервера 6 vCPU / 12 GB RAM.

---

## [0.5.6] — 2026-03

### Тарифы с токенами и лимитами
- **Backend:** Free — 1 мастеринг в неделю (IP); Pro — 50 токенов/мес, до 10 в день; Studio — 100 токенов/мес, до 30 в день. Баланс токенов в БД (`tokens_balance`), списание при мастеринге, доплата токенов через YooKassa. `deps.py`: недельный лимит для Free; `mastering.py`: проверка и списание для Pro/Studio; `payments.py`: планы и webhook с начислением токенов; `misc.py`: `/api/limits` возвращает `tokens_balance`, `daily_used`, `daily_limit`, `remaining`.
- **Frontend:** pricing.html и локали — тексты тарифов (1 в неделю, 50/100 токенов, лимиты в день); убрана «Поддержка 24/7» из Studio. В шапке отображается баланс токенов для Pro/Studio; блокировка кнопки мастеринга при `remaining === 0`.

### Мобильная шапка
- **frontend/index.html:** реструктуризация шапки — `header-top-row`, `logo-inner`; медиа-запросы 640/480/360px: отступы, скрытие подписей темы/языка, увеличенные touch targets; кнопки больше не наезжают друг на друга.

### Лимиты загрузки
- **Backend:** лимиты по формату (WAV 800 МБ, MP3 300 МБ, FLAC 500 МБ) и по тарифу (Free 100, Pro 300, Studio 800 МБ). `config.py`, `settings_store.py`: `get_max_upload_mb(filename, tier)`; все эндпоинты мастеринга/анализа/measure/isolate-vocal используют его. Админка: поля `max_upload_mb_wav`, `max_upload_mb_mp3`, `max_upload_mb_flac`, `max_upload_mb_free/pro/studio`.
- **doc/UPLOAD_LIMITS.md:** описание лимитов и рекомендации по RAM для больших файлов (DJ-сеты).

---

## [0.5.5] — 2026-03

### Лендинг (hero)
- **frontend/landing.html**: блок hero — зона загрузки файла заменена на одну CTA-кнопку «Войти в приложение»; переверстана под стиль страницы (размер кнопки, типографика, градиент, тень, hover); удалён блок hero-note («Нет трека? … Смотреть тарифы»); удалены неиспользуемые стили.

---

## [0.5.4] — 2026-03

### Дополнительная обработка (план 12.2)
- **frontend/app.js**: при включённом Transient Designer и ползунках 100/100 отправляются 1.02/0.98, чтобы эффект применялся; для Parallel Compression передаётся mix не ниже 0.01.
- **backend/app/routers/mastering.py**: надёжное приведение `parallel_mix` к float при применении Parallel Compression.
- **backend/tests/test_api.py**: тест `test_api_v2_master_accepts_pro_params` — POST /api/v2/master принимает все PRO-параметры (rumble, denoiser, deesser, transient, parallel, dynamic_eq).
- **backend/tests/test_e2e_mastering.py**: тест `test_e2e_mastering_with_pro_rumble` — E2E мастеринг с румбл-фильтром до получения результата.
- **backend/tests/test_pipeline.py**: тест `test_pro_modules_transient_parallel_dyn_eq_not_silent` — Transient Designer, Parallel Compression и Dynamic EQ не дают тишину на выходе.

---

## [0.5.3] — 2026-03

### Админка и настройки
- **backend/app/admin.py**: лимиты AI допускают -1 (без лимита); SMTP port ge=0; llm_guard min 1; при port=0 сохраняется 587; ослаблена валидация для устранения 422 при сохранении.
- **backend/app/ai.py**: ключи OpenAI/DeepSeek читаются только из настроек админки (`_get_llm_setting_admin_only`), не из .env.
- **frontend/admin.html**: buildSmtpPayload — порт в диапазоне 1–65535, иначе 587.

### UI: волновая форма на весь экран
- **frontend/index.html**: полноэкранный оверлей волновой формы (кнопка «Развернуть», двойной клик по графику); кнопка Play/Pause в оверлее; цветовая схема как у блока «Спектр» (фон, заголовок, область графика, светлая тема).
- **frontend/app.js**: drawWaveformToCanvas(), открытие/закрытие оверлея, синхронизация кнопки воспроизведения с основным плеером.

### Тесты
- **backend/tests/test_api.py**: тест `test_api_v2_chain_default_style_dry_vocal`; тест `test_api_v2_master_unknown_style_fallback`.

---

## [0.5.2] — 2026-03

### Тесты
- **backend/tests/test_api.py**: тест `test_api_measure_returns_lufs` — POST /api/measure возвращает lufs, sample_rate, peak_dbfs, duration, channels; тест `test_api_measure_rejects_bad_extension` — POST /api/measure с неподдерживаемым расширением → 400; тест `test_api_v2_analyze_extended` — POST /api/v2/analyze с extended=true проверяет опциональные поля spectrum_bars, lufs_timeline; тест `test_api_master_status_structure` — GET /api/master/status/{job_id} возвращает status, progress, message и корректные типы; тест `test_api_master_result_404` — GET /api/master/result/{job_id} с несуществующим id → 404; тест `test_api_styles_each_has_lufs` — GET /api/styles: у каждого стиля есть ключ lufs (число).

---

## [0.5.1] — 2026-03

### Расширение тестов
- **backend/tests/test_api.py**: тесты для `POST /api/v2/isolate-vocal` (503/400); `test_api_health_returns_features` (поля features в GET /api/health); `test_api_v2_batch_with_one_file_creates_job` (batch с одним файлом → jobs с job_id); расширена проверка GET /api/v2/chain/default (version, style, target_lufs, у каждого модуля id и label).
- **backend/tests/test_pipeline.py**: тест `test_run_mastering_pipeline_style_dry_vocal` — пайплайн со стилем dry_vocal без NaN, корректная форма выхода.
- **backend/tests/test_ai.py**: тест `test_report_recommendations_have_expected_structure` — проверка структуры рекомендаций в отчёте (строки или объекты с текстом).

### Рефакторинг архитектуры (2026-03-01)

#### Деплой без Docker — systemd + Nginx
- **deploy/systemd/magic-master.service**: unit-файл для запуска через systemd с лимитами памяти (MemoryMax=10G), CPU affinity и security directives.
- **deploy/deploy.sh**: скрипт автоматического деплоя на Ubuntu 22.04 (пакеты, venv, systemd, Nginx, log rotation).
- **deploy/nginx/magic-master.conf**: upstream переключён на 127.0.0.1:8000, добавлен keepalive 32, gzip-компрессия.
- **deploy/nginx/magic-master-proxy.inc**: выделенный include с proxy-настройками, SSE-локейшном и security headers.

#### Разбивка main.py на роутеры
- **backend/app/routers/misc.py**: публичные вспомогательные эндпоинты (`/api/news`, `/api/limits`, `/api/presets`, `/api/styles`, `/api/measure` и др.).
- **backend/app/routers/ai_router.py**: все AI-эндпоинты (`/api/ai/limits`, `/api/ai/recommend`, `/api/ai/report`, `/api/ai/nl-config`, `/api/ai/chat`).
- **backend/app/routers/auth.py**: аутентификация и профиль пользователя (register, login, me, profile, history, API-ключи, пресеты, смена/сброс пароля).
- **backend/app/routers/mastering.py**: весь цикл мастеринга (POST /api/v2/master, /api/v2/batch, /api/v2/analyze, /api/v2/reference-match, статус, результат, прогресс SSE).
- **backend/app/routers/__init__.py**: маркер пакета.
- **backend/app/main.py**: сокращён с ~2986 до ~560 строк — только инициализация, middleware, health, metrics, статика.

#### Слой сервисов для admin.py
- **backend/app/services/user_service.py**: бизнес-логика управления пользователями (CRUD, подписки, bulk-действия).
- **backend/app/services/stats_service.py**: агрегация статистики для дашборда.
- **backend/app/services/reports_service.py**: генерация аналитических отчётов.
- **backend/app/services/__init__.py**: маркер пакета.
- **backend/app/admin.py**: упрощён до «тонкого» контроллера, вызывающего сервисы.

#### Общие зависимости и утилиты
- **backend/app/deps.py**: централизованные FastAPI-зависимости и rate-limit хранилища (`_rate_limits`, `_auth_attempts`, `_FREE_DAILY_LIMIT`, `_AUTH_LIMIT_PER_MINUTE`).
- **backend/app/helpers.py**: добавлены `json_safe_float` и `safe_content_disposition_filename`.
- **backend/app/jobs_store.py**: SQLite-персистентность задач мастеринга с in-memory кэшем, семафоры для очереди приоритетов.

#### Оптимизация DSP
- **backend/app/pipeline.py**: JIT-компиляция через `numba` (`_envelope_follower_core`, `_comb_filter_core`, `_allpass_filter_core`).
- **backend/requirements.txt**: добавлен `numba>=0.59.0`.

#### Исправление тестов
- **backend/tests/test_api.py**, **test_auth.py**, **test_admin.py**: импорты `_rate_limits`, `_auth_attempts`, `_AUTH_LIMIT_PER_MINUTE`, `_FREE_DAILY_LIMIT`, `_check_audio_magic_bytes`, `_reset_tokens` перенаправлены из `app.main` в `app.deps`, `app.helpers`, `app.routers.auth`.

### Расширение i18n (бэклог)

- **frontend/locales/ru.json**, **en.json**: добавлены ключи для шапки (подзаголовок, тема, тёмная/светлая), тир-бейджа (мастеринга осталось, Перейти на Pro), авторизации (Профиль, Дашборд, Выйти, Войти/Регистрация, Приоритетная очередь), зоны загрузки (перетащите файл, или нажмите), пакетной обработки, параметров (Параметры, Жанр/Стиль, Целевой LUFS, Цепочка модулей, Доп. обработка), кнопок (Сохранить/Загрузить/Удалить пресет, Запустить мастеринг, Обработка…, файлов), A/B (Прослушать до/после, Сравнение до/после).
- **frontend/index.html**: атрибуты `data-i18n` проставлены на соответствующие элементы; при переключении RU/EN обновляется весь интерфейс.
- **frontend/app.js**: текст кнопки мастеринга («Запустить мастеринг», «Обработка…») выставляется через `__t('app.master')` и `__t('app.processing')` с учётом текущей локали.

### Очередь 3.4 — Расширение пресетов сообщества (2026-03)

- **backend/app/presets_community.json**: добавлены пресеты Rock (−12 LUFS, standard), Jazz (−18 LUFS, classical), Cinematic (−24 LUFS, standard); для пресета «Подкаст» указан стиль `podcast`.
- **backend/app/PRESETS_COMMUNITY_README.md**: описание формата полей (id, name, target_lufs, style, chain_config), список допустимых `style`, инструкция по добавлению пресетов.

### Очередь 3.3 — Доп. алерты мониторинга (2026-03)

- **backend/app/config.py**: добавлены `alert_monitoring_enabled`, `alert_queue_threshold`, `alert_throttle_minutes` (MAGIC_MASTER_ALERT_*).
- **backend/app/notifier.py**: функции `notify_alert_health_degraded(reason, details)` и `notify_alert_queue_threshold(jobs_total, jobs_running)` с троттлингом (не чаще раза в N минут). Настройки читаются из settings_store (админка) с fallback на config.
- **backend/app/main.py**: при ответе `/api/health` при статусе degraded вызывается `notify_alert_health_degraded`; при ответе `/api/metrics` при превышении порога очереди — `notify_alert_queue_threshold`. Уведомления уходят в Telegram (те же токен/chat_id, что и для прочих алертов).
- **backend/app/settings_store.py**, **admin.py**: ключи алертов добавлены в хранилище и в API настроек (GET/PATCH). В админке в разделе «Настройки» → «Общие» добавлен блок «Алерты мониторинга» (чекбокс включения, порог очереди, интервал повтора).

### Очередь 1.2–1.3 — ROADMAP и синхронизация документации (2026-03)

- **Очередь 1.2:** В ROADMAP.md пункт «ITU/E-weighted опции в API» уже отмечен выполненным `[x]`; в плане доработок задача 1.2 отмечена как выполненная.
- **Очередь 1.3:** Обновлены даты в PROGRESS.md и ROADMAP.md (2026-03); в PROGRESS уточнён блок «Осталось задач к разработке» (состояние очередей 1–4). doc/ФУНКЦИИ_И_УРОВНИ.md и doc/PLAN_DORABOTKI.md приведены в соответствие (ссылки и даты).

### Очередь 2.3 — Отдельный пункт «LLM» в сайдбаре (2026-03)

- **frontend/admin.html**: в сайдбар добавлена группа «LLM» с подпунктами «Подключение» и «Промпты». Страница «Подключение (LLM)» — форма бэкенда, API-ключей, лимитов и защиты от инъекций; страница «Промпты (LLM)» — список промптов с редактированием, историей и восстановлением. Вкладка «LLM» удалена из «Настройки»; в Настройках остались только Общие, SMTP, YooKassa, Telegram. Обработчики модалок промптов вызывают `refreshPromptsBlock()` для обновления текущей страницы (LLM Промпты или Настройки).

### Очередь 4.3 — Плагины/расширения (2026-03)

- **backend/app/config.py**: добавлена настройка `community_presets_extra` (MAGIC_MASTER_COMMUNITY_PRESETS_EXTRA) — путь к дополнительному JSON-файлу или каталогу с пресетами сообщества.
- **backend/app/routers/misc.py**: загрузка пресетов расширена: после `presets_community.json` подгружаются пресеты из extra (файл или все .json в каталоге), дубликаты по `id` отбрасываются. Добавлен GET /api/extensions — минимальный API статуса расширений (community_presets_extra_configured, community_presets_extra_loaded).
- **doc/PLUGINS_EXTENSIONS.md**: описание архитектуры расширений (пресеты сообщества, задел под будущие процессоры).
- **backend/app/PRESETS_COMMUNITY_README.md**: упоминание переменной COMMUNITY_PRESETS_EXTRA и ссылка на doc/PLUGINS_EXTENSIONS.md.
- **backend/tests/test_api.py**: тест test_api_extensions для GET /api/extensions.

### Очередь 4.2 — Отчёт «Рекомендации по промптам» (2026-03)

- **backend/app/services/reports_service.py**: добавлен отчёт `prompt_recommendations` в REPORTS_META. Реализация: по каждому типу промпта (recommend, report, nl_config, chat) выводится число использований из `ai_usage_log` за период и превью активного текста промпта из БД. Кнопка «Резюме LLM» в админке формирует краткие рекомендации по данным отчёта.

### Очередь 4.1 — Именованные шаблоны промптов (2026-03)

- **backend/app/admin.py**: в ответах GET /api/admin/prompts и GET /api/admin/prompts/{slug}/history в каждую версию добавлено поле `name` (имя шаблона или «v{version}»). Добавлен GET /api/admin/prompts/{slug}/version/{version_id} — возвращает { id, version, name, body } для кнопки «Применить шаблон».
- **frontend/admin.html**: в модальном окне редактирования промпта добавлены блок «Применить шаблон» (выпадающий список версий + кнопка «Применить»), загрузка тела выбранной версии через новый API, и необязательное поле «Название версии» при сохранении (передаётся как save_as_template_name).

### Очередь 3.5 — Улучшение тестов (2026-03)

- **backend/tests/test_api.py**: тест `test_api_v2_master_accepts_bitrate` — POST /api/v2/master с параметром bitrate (при out_format=wav битрейт игнорируется); тест `test_api_v2_batch_requires_files` — POST /api/v2/batch без файлов возвращает 400/422/503.
- **backend/tests/test_pipeline.py**: тесты `test_export_audio_mp3_with_bitrate` и `test_export_audio_opus_with_bitrate` — проверка экспорта MP3 (128/192/320) и OPUS (128/192) с параметром bitrate (пропуск при отсутствии ffmpeg).

### Очередь 3.2 — Доп. битрейты MP3/OPUS (2026-03)

- **backend/app/pipeline.py**: в `export_audio()` добавлен параметр `bitrate` (опционально); для MP3 используются 128/192/256/320 kbps, для OPUS — 128/192 kbps (по умолчанию 320 и 192).
- **backend/app/routers/mastering.py**: в формах v2/master, v2/batch и v2/reference-match добавлен параметр `bitrate`; добавлена `_normalize_bitrate()` для проверки допустимых значений; битрейт передаётся в `export_audio`.
- **frontend/index.html**: блок «Битрейт (kbps)» с выпадающим списком (id="outBitrateWrap", "outBitrate"), отображается при выборе MP3 или OPUS.
- **frontend/app.js**: при смене формата обновляются опции битрейта (MP3: 128/192/256/320, OPUS: 128/192); при отправке форм мастеринга, batch и reference-match в запрос добавляется `bitrate`.

### Очередь 3.1 — Расширение i18n (2026-03)

- **frontend/locales/ru.json**, **en.json**: ключи для эталона (ref_track, ref_not_selected, ref_intensity), экспорта (export_section, export_dither, export_auto_blank), форматов (format_wav, format_mp3, format_flac, format_opus, format_aac), блока «Доп. обработка» (pro_preset, pro_strength, pro_threshold, pro_attack, pro_sustain, pro_mix, denoiser_custom/light/medium/aggressive).
- **frontend/index.html**: атрибуты data-i18n на подписи эталона, экспорта, форматов и опций дизеринга/обрезки, пресета деноайзера и подписей слайдеров (Пресет, Сила, Порог, Атака, Сустейн, Mix).
- **frontend/app.js**: плейсхолдер «не выбран» для эталона выставляется через __t('app.ref_not_selected'); при смене языка applyI18n не перезаписывает имя файла эталона (window.__refFile).

### Очередь 2 — Админка: отчётность (2026-03)

- **frontend/admin.html**: На странице «Отчётность» добавлена сортировка по колонкам таблицы (клик по заголовку — сортировка по колонке, повторный клик переключает направление; числовые и строковые значения). При загрузке списка отчётов автоматически выбирается первый отчёт (класс `active`), при отсутствии отчётов выводится подсказка «Нет доступных отчётов». Стили для `.report-th-sort` (курсор, hover, индикаторы ▲/▼).

### Единый план доработок

- **doc/PLAN_DORABOTKI.md**: единый план доработок в четыре очереди — 1) закрепление рефакторинга и деплоя в git, синхронизация ROADMAP; 2) админка (сортировка по колонкам в отчётности, подсветка выбранного отчёта, опционально пункт LLM в сайдбаре); 3) бэклог продукта (i18n, доп. битрейты, алерты, пресеты, тесты); 4) опции (шаблоны промптов, отчёт по промптам, плагины). PROGRESS.md, ROADMAP.md, ФУНКЦИИ_И_УРОВНИ.md, AUDIT_ADMIN_REDESIGN.md увязаны со ссылками на этот план; в ROADMAP отмечено выполнение ITU/E-weighted dither.

### Точечные доработки (аудит и валидация)

- **backend/app/admin.py**: валидация PATCH настроек через Pydantic `Field` — `max_upload_mb` (1–500), `default_target_lufs` (−60…−1), `jobs_done_ttl_seconds` (0–30 дней), `global_rate_limit` (1–10000), `ai_limit_*`, `llm_guard_max_length_*`, SMTP `port` (1–65535). Согласованность с ограничениями на фронте.
- **doc/AUDIT_ADMIN_REDESIGN.md**: раздел 1.3 приведён в соответствие с реализацией (maintenance_mode, флаги функций, журнал, валидация настроек отмечены как выполненные).

---

## [0.5.0] — 2026-03

### Очередь 9 — Обработка вокала (полностью закрыта)

- **Стиль dry_vocal:** пресет с ровной АЧХ, без эксайтера и параллельной компрессии; карточка в блоке стилей, локализация, дашборд.
- **Пресеты Spectral Denoiser:** tape_hiss (ленточный шип), room_tone (тон комнаты); UI, API, локализация, тесты, doc/PIPELINE_AUDIO_QUALITY.md.
- **Румбл-фильтр (9.1):** high-pass 80 Гц в блоке «Доп. обработка» — `backend/app/pipeline.py` (`apply_rumble_filter`), роутер, UI (чекбокс + срез 60–120 Гц), тест `test_apply_rumble_filter`.
- **Полоса де-эссера 5–8 кГц (9.3):** выбор «5–9 кГц» / «5–8 кГц (вокал)» в UI, параметр `deesser_freq_hi` в API и пайплайне.
- **Пакетный мастеринг (batch):** передача PRO-параметров (румбл, Denoiser, De-esser, Transient, Parallel, Dynamic EQ) — те же настройки «Доп. обработки» для всех файлов в пакете.
- **Изоляция вокала (9.2):** опциональная фича на базе Demucs: `backend/run_isolate_vocal.py`, `backend/requirements-vocal-isolation.txt`, `backend/app/services/vocal_isolation.py`, эндпоинт `POST /api/v2/isolate-vocal`, конфиг `MAGIC_MASTER_ENABLE_VOCAL_ISOLATION`, `MAGIC_MASTER_DEMUCS_MODEL`; интеграция в пайплайн (опция «Сначала изолировать вокал») и блок в UI «Доп. обработка» (виден при `vocal_isolation_enabled` в /api/health). Документация: doc/PLAN_9_2_VOCAL_ISOLATION.md.

### Версионность

- **doc/VERSIONING.md:** правила версионности (SemVer, где хранится версия, когда менять MINOR/PATCH, связь с CHANGELOG).
- Версия приложения: единственный источник — `backend/app/version.py`; отображается в /api/version, /api/health, в футере интерфейса.

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
