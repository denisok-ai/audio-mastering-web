# Прогресс выполнения плана — Magic Master

> Обновлено: 2026-03 (очереди 1–10 закрыты: в т.ч. 9.2 изоляция вокала, 10.1–10.5 аудит фронт/бэк)  
> В интерфейсе: футер → **Статус плана** (открывает этот файл). API: `GET /api/progress`

---

## Процент выполнения плана разработки

| План | Выполнено | Всего | Процент |
|------|-----------|-------|---------|
| **Фазы P0–P56** (основной план) | 30 | 30 | **100%** |
| **Бэклог P57+** (по запросу) | 5 | 5 | **100%** |
| **Рефакторинг (подпункты)** | 4 | 4 | **100%** |

**Итого: план разработки выполнен на 100%.** Основной план (P0–P56), бэклог (P57–P59, P64 + рефакторинг P60–P63) и все подпункты рефакторинга закрыты. Последний пункт бэклога («прочие идеи по запросу») не содержит конкретных задач — новые фичи по запросу.

### Осталось задач к разработке

| Категория | Осталось | Описание |
|-----------|----------|----------|
| **Единый план доработок** | см. [doc/PLAN_DORABOTKI.md](doc/PLAN_DORABOTKI.md) | Очереди 1–10 выполнены (9.2 изоляция вокала; 10.1–10.5 тарифы с бэка, кнопка «Только изолировать вокал», документирование API). Дальнейшие задачи — по запросу (расширение тестов, идеи из PROGRESS или фидбека). |
| **План админ-панели** | см. [doc/PLAN_DORABOTKI_ADMIN.md](doc/PLAN_DORABOTKI_ADMIN.md) | Очереди 1–3 выполнены (выход, API-ключи, настройки). |
| **План PRO-модулей** | см. [doc/PLAN_DORABOTKI_PRO_MODULES.md](doc/PLAN_DORABOTKI_PRO_MODULES.md) | Тишина, De-esser, Denoiser (Vocal), Dynamic EQ, диагностика — выполнены. |
| **План ребрендинга** | см. [doc/PLAN_DORABOTKI_REBRAND_MODULES.md](doc/PLAN_DORABOTKI_REBRAND_MODULES.md) | Упоминания сторонних плагинов убраны (п. 1–7). |
| **Бэклог P57–P64** | **0** | Все пункты реализованы. |

**Итого:** основной план (P0–P64) и план доработок (очереди 1–10) выполнены. Продукт готов к эксплуатации.

---

## Режим: пауза разработки новых фич

**Текущая версия считается готовой к эксплуатации.** Разработка новых фич поставлена на паузу. Реализовано 56 фаз (P0–P56): полный цикл мастеринга, тарифы, оплата, админка, AI, PWA, безопасность (CORS, webhook). Оставшиеся идеи из плана (доп. форматы экспорта, мониторинг, i18n, плагины) переведены в **бэклог** и подключаются по запросу. Фокус — эксплуатация, мониторинг и точечные доработки по инцидентам. См. [doc/RUNBOOK.md](doc/RUNBOOK.md) для действий при сбоях.

---

## Статус доработок (кратко)

| Область | Статус | Детали |
|---------|--------|--------|
| **AI-агенты** | Готово | Рекомендатор пресета, отчёт по анализу, авто-мастеринг, NL→настройки, чат. Лимиты по тарифам (Free 5/день, Pro 50, Studio без лимита). |
| **Бэкенды LLM** | Готово | OpenAI (gpt-4o-mini) и **DeepSeek** (API совместим с OpenAI). Переменные: `MAGIC_MASTER_AI_BACKEND=deepseek`, `DEEPSEEK_API_KEY`. |
| **Конфиг AI** | Готово | config.py: ai_backend, openai_api_key, deepseek_api_key, deepseek_base_url, deepseek_model, ai_limit_free/pro/studio. |
| **API AI** | Готово | GET /api/ai/limits, POST /api/ai/recommend, POST /api/ai/report, POST /api/v2/master/auto, POST /api/ai/nl-config, POST /api/ai/chat. |
| **UI для AI** | Готово | Кнопки «AI рекомендует», «Авто-мастеринг» в параметрах; кнопка «AI отчёт» после замера (P17); отображение ai_backend и остатка запросов. |
| **Баги и стабильность** | Готово | Spectral Denoiser (min_gain, защита от обнуления), векторскоп/спектр при A/B, favicon 204, БД create_tables (already exists), PROGRESS.md в Docker. |
| **Админ-панель** | Готово | P18-P23: БД расширена, Admin API, Admin SPA, новости на лендинге, SMTP рассылки, YooKassa. |

---

> Обновлено: 2026-03

## Сводка по приоритетам

| Приоритет | Статус | Выполнено | Осталось |
|-----------|--------|-----------|----------|
| **P0** — Текущий спринт (DAW UI, багфиксы) | ✅ Завершён | 7/7 | 0 |
| **P1** — Аудит-фиксы | ✅ Завершён | 5/5 | 0 |
| **P2** — Модульный бэкенд v2 | ✅ Завершён | 4/4 + API v2 | 0 |
| **P3** — Расширение процессоров | ✅ Завершён | EQ linear phase, M/S per-band, 4-band Imager, Reverb M/S, Dithering, analyze, oversampling | 0 |
| **P4** — Frontend v2 | ✅ Завершён | app.js, styles.css, спектр, векторскоп, цепочка, drag-and-drop, Amount, phase_mode, correlation | 0 |
| **P5** — Портал + тарифы | ✅ Завершён | landing.html, pricing.html, rate limiting, tier locks, upgrade modal, /api/limits | 0 |
| **P6** — Аутентификация | ✅ Завершён | SQLite+SQLAlchemy, bcrypt, JWT, register/login/me, auth UI | 0 |
| **P7** — История + Dashboard | ✅ Завершён | MasteringRecord, history API, dashboard.html, change-password | 0 |
| **P8** — Режим отладки | ✅ Завершён | MAGIC_MASTER_DEBUG, /api/debug-mode, bypass лимитов, UI-бейдж | 0 |
| **P9** — Пакетная обработка (Batch) | ✅ Завершён | POST /api/v2/batch, UI выбора нескольких файлов, прогресс и скачивание | 0 |
| **P10** — Сохранение пресетов цепочки | ✅ Завершён | SavedPreset в БД, GET/POST/DELETE /api/auth/presets, UI «Сохранить/Загрузить/Удалить» в карточке цепочки | 0 |
| **P11** — Мобильная адаптация | ✅ Завершён | Breakpoints 640px/480px/360px, touch targets (pointer: coarse), safe-area, цепочка/пресеты/тосты на малых экранах | 0 |
| **P12** — A/B с отображением LUFS | ✅ Завершён | LUFS оригинала и мастера рядом с кнопками A/B в плеере; сброс при смене файла/очистке | 0 |
| **P13** — Экспорт отчёта анализа | ✅ Завершён | Кнопка «Скачать отчёт» после замера; текстовый отчёт (LUFS, peak, длительность, спектр, LUFS по времени) | 0 |
| **P14** — Приоритетная очередь для Pro | ✅ Завершён | Семафоры: 2 слота для Pro/Studio, 1 для Free; GET /api/limits → priority_queue; бейдж «Приоритетная очередь» в шапке | 0 |
| **P15** — Экспорт отчёта в JSON | ✅ Завершён | Кнопка «JSON» рядом с «Скачать отчёт»; скачивание полного результата анализа в .json | 0 |
| **P16** — AI-агенты и тарифы | ✅ Завершён | Рекомендатор, отчёт, авто-мастеринг, NL→config, чат; лимиты по тарифам; бэкенды OpenAI и DeepSeek | 0 |
| **P17** — AI отчёт в UI | ✅ Завершён | Кнопка «AI отчёт» рядом с Скачать отчёт/JSON; вызов /api/ai/report, блок с summary и списком рекомендаций | 0 |
| **P18** — Расширение БД + конфиг | ✅ Завершён | is_admin/is_blocked/subscription в User; модели Transaction, NewsPost, EmailCampaign; SMTP и YooKassa в config.py | 0 |
| **P19** — Admin API | ✅ Завершён | /api/admin/*: stats, users, transactions, news, campaigns; GET /api/news (публичный) | 0 |
| **P20** — Admin UI | ✅ Завершён | frontend/admin.html: Dashboard, Пользователи, Транзакции, Новости, Рассылки; GET /admin | 0 |
| **P21** — Новости на лендинге | ✅ Завершён | Секция «Новости» на landing.html, fetch к /api/news | 0 |
| **P22** — SMTP рассылки | ✅ Завершён | mailer.py: send_email, send_welcome_email, send_subscription_activated_email | 0 |
| **P23** — YooKassa | ✅ Завершён | payments.py: POST /api/payments/create, webhook; кнопки оплаты в pricing.html | 0 |
| **P24** — AI Chat UI | ✅ Завершён | Плавающая кнопка + чат-панель с историей, POST /api/ai/chat с контекстом анализа, для Pro/Studio | 0 |
| **P25** — NL→настройки в UI | ✅ Завершён | Поле в блоке цепочки, POST /api/ai/nl-config → подстановка chain_config и target_lufs | 0 |
| **P26** — pytest Admin+Payments | ✅ Завершён | tests/test_admin.py: 17 тестов (auth guard, users CRUD, subscription, news, campaigns, payments webhook) | 0 |
| **P54** — pytest pipeline и AI | ✅ Завершён | test_pipeline.py (цепочка, экспорт, анализ), test_ai.py (лимиты, рекомендатор, отчёт, nl_to_config). | 0 |
| **P55** — Spectral Denoiser пресеты | ✅ Завершён | DENOISE_PRESETS (light/medium/aggressive), denoise_preset + denoise_noise_percentile в API, селект в UI. | 0 |
| **P56** — CORS и webhook | ✅ Завершён | Настраиваемый CORS_ORIGINS; IP whitelist для webhook YooKassa; фикс notify_payment в webhook. | 0 |
| **P57** — Экспорт AAC (M4A) | ✅ Завершён | pipeline: export_audio("aac") → M4A 192 kbps; main: aac во всех эндпоинтах, расширение .m4a; UI: опция AAC в селекте. | 0 |
| **P58** — Метрики для мониторинга | ✅ Завершён | GET /api/metrics: uptime_seconds, jobs_running, jobs_total, version; RUNBOOK и тест. | 0 |
| **P59** — Базовая i18n (локализация) | ✅ Завершён | GET /api/locale; /locales/ru.json, en.json; переключатель RU/EN, data-i18n, __t(). | 0 |
| **P60** — Проверка magic bytes при загрузке | ✅ Завершён | _check_audio_magic_bytes (WAV/FLAC/MP3); вызов во всех эндпоинтах загрузки; тест. | 0 |
| **P61** — Доступность (a11y) | ✅ Завершён | aria-label на кнопках (сброс, play/pause, эталон, чат); role="status" и aria-live для прогресса и тостов; :focus-visible. | 0 |
| **P62** — Вынос хелперов из main.py | ✅ Завершён | app/helpers.py: get_client_ip, allowed_file, check_audio_magic_bytes; main импортирует алиасами. | 0 |
| **P63** — E2E-тест мастеринга | ✅ Завершён | test_e2e_mastering.py: загрузка WAV → POST v2/master → опрос status → GET result, проверка RIFF. | 0 |
| **P64** — Пресеты сообщества | ✅ Завершён | GET /api/presets/community; presets_community.json; UI: optgroup в селекте, загрузка без логина. | 0 |

---

## Фаза P64 — Пресеты сообщества ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| backend/app/presets_community.json | ✅ | Три пресета: Stream (−14), Подкаст (−16), Клуб (−9). Формат: id, name, target_lufs, style, chain_config. |
| main.py | ✅ | _load_community_presets(), GET /api/presets/community. |
| frontend | ✅ | loadSavedPresetsList: запрос к /api/presets/community, optgroup «Пресеты сообщества»; Load применяет target_lufs/style; кнопка «Удалить» отключена для пресетов сообщества. |
| test_api.py | ✅ | test_api_presets_community. |

---

## Фаза P63 — E2E-тест мастеринга ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| tests/test_e2e_mastering.py | ✅ | test_e2e_mastering_flow: POST /api/v2/master с minimal_wav_bytes, опрос /api/master/status/{job_id}, GET result, проверка WAV (RIFF). Использует фикстуру minimal_wav_bytes из conftest. |

---

## Фаза P62 — Вынос хелперов из main.py ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| backend/app/helpers.py | ✅ | Новый модуль: get_client_ip(Request), allowed_file(filename), check_audio_magic_bytes(data, filename). Зависимость: config.settings. |
| backend/app/main.py | ✅ | Импорт из .helpers с алиасами _get_client_ip, _allowed_file, _check_audio_magic_bytes; удалены локальные определения. Тесты без изменений (импорт из main). |

---

## Фаза P61 — Доступность (a11y) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| index.html | ✅ | aria-label на btnReset, btnPP, btnRefClear, chatSend; role="status" aria-live="polite" на progWrap; aria-live на toastWrap. |
| CSS | ✅ | button:focus-visible, a:focus-visible — видимый outline для навигации с клавиатуры. |

---

## Фаза P60 — Проверка magic bytes при загрузке ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| main.py | ✅ | _check_audio_magic_bytes(data, filename): WAV (RIFF…WAVE), FLAC (fLaC), MP3 (ID3 или 0xFF 0xE?). |
| Эндпоинты | ✅ | Проверка после чтения файла: /api/measure, /api/master, /api/v2/master, /api/v2/batch, /api/v2/master/auto, /api/v2/analyze, /api/v2/reference-match, AI recommend/report. |
| test_api.py | ✅ | test_check_audio_magic_bytes — валидные и невалидные сигнатуры. |

---

## Фаза P59 — Базовая i18n (локализация) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| backend | ✅ | GET /api/locale — available: ["ru","en"], default: "ru". |
| frontend/locales | ✅ | ru.json, en.json — строки app.title, app.measure, app.upload, app.download_report, locale.label и др. |
| index.html | ✅ | Переключатель языка (RU/EN) в шапке; data-i18n на заголовок, кнопки, карточку загрузки. |
| app.js | ✅ | getLocale/setLocale (localStorage + ?lang=), loadLocale(), applyI18n(), привязка к кнопкам. |
| test_api.py | ✅ | test_api_locale проверяет ответ /api/locale. |

---

## Фаза P58 — Метрики для мониторинга ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| main.py | ✅ | GET /api/metrics — плоский JSON для скрапинга (uptime_seconds, jobs_running, jobs_total, version). |
| test_api.py | ✅ | test_api_metrics проверяет наличие полей и типов. |
| doc/RUNBOOK.md | ✅ | Упоминание /api/metrics в разделе проверки состояния. |

---

## Фаза P57 — Экспорт AAC (M4A) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| pipeline.py | ✅ | export_audio для out_format="aac" (pydub ipod + aac 192 kbps, ffmpeg). |
| main.py | ✅ | Разрешён формат aac в POST /api/master, /api/v2/master, /api/v2/batch, /api/v2/master/auto, ref-match; имя файла _mastered.m4a; _mime_map для preview. |
| frontend | ✅ | Опция «AAC — 192 kbps (M4A) 🔒» в селекте формата; расширение .m4a при скачивании (app.js). |
| test_pipeline.py | ✅ | test_export_audio_aac_returns_bytes (пропуск без ffmpeg). |

---

## Фаза 17 — AI отчёт в UI (P17) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Кнопка «AI отчёт» | ✅ | В блоке reportActions после «Скачать отчёт» и «JSON» |
| Вызов POST /api/ai/report | ✅ | С body.analysis (lastAnalyzeReport) или с file, если замера не было |
| Блок с результатом | ✅ | aiReportResult: summary + список recommendations под кнопками; показ и плавный скролл после ответа |
| Обновление лимитов | ✅ | loadAiLimits() после успешного запроса |

---

## Фаза 16 — AI-агенты и тарифы (P16) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Конфиг AI | ✅ | config.py: ai_backend, openai/deepseek ключи, лимиты по тарифам |
| Модуль ai.py | ✅ | recommend_preset, report_with_recommendations, nl_to_config, chat_assistant; _get_llm_client() для OpenAI/DeepSeek |
| GET /api/ai/limits | ✅ | Лимиты AI (ai_used, ai_limit, ai_remaining, ai_backend) |
| POST /api/ai/recommend | ✅ | Рекомендация пресета по анализу (file или body.analysis) |
| POST /api/ai/report | ✅ | Текстовый отчёт + рекомендации по анализу |
| POST /api/v2/master/auto | ✅ | Анализ → рекомендатор → мастеринг одним запросом |
| POST /api/ai/nl-config | ✅ | Преобразование запроса на естественном языке в chain_config/target_lufs |
| POST /api/ai/chat | ✅ | Чат-помощник по мастерингу с контекстом |
| DeepSeek | ✅ | ai_backend=deepseek, DEEPSEEK_API_KEY, base_url, model; общий клиент LLM |
| README | ✅ | Переменные окружения для AI и инструкция по DeepSeek |

---

## Фаза 15 — Экспорт отчёта в JSON (P15) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Кнопка «JSON» | ✅ | Рядом с «Скачать отчёт»; видна после замера (блок reportActions) |
| Скачивание JSON | ✅ | JSON.stringify(lastAnalyzeReport, null, 2), файл {имя}_analyze_report.json |

---

## Фаза 14 — Приоритетная очередь для Pro (P14) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Семафоры слотов | ✅ | Процесс мастеринга занимает слот: Pro/Studio (и debug) — 2 одновременных задачи, Free — 1 |
| POST /api/master и POST /api/v2/master | ✅ | Задача выполняется в async with sem (приоритетный или обычный семафор) |
| POST /api/v2/batch | ✅ | Каждый файл в пакете использует тот же семафор (Pro — приоритетный) |
| GET /api/limits | ✅ | В ответ добавлено поле priority_queue: true для Pro/Studio/debug, false для Free |
| UI: бейдж «Приоритетная очередь» | ✅ | В шапке для залогиненных Pro показывается подсказка (при priority_queue === true) |

---

## Фаза 13 — Экспорт отчёта анализа (P13) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Сохранение результата расширенного анализа | ✅ | lastAnalyzeReport заполняется после «Измерить громкость», сбрасывается при смене файла/сбросе |
| Кнопка «Скачать отчёт» | ✅ | Отображается после замера, скачивает .txt с именем файла_analyze_report.txt |
| Содержимое отчёта | ✅ | Файл, дата, LUFS, Peak dBFS, длительность, sample rate, каналы, стерео-корреляция; при наличии — мин/макс/среднее LUFS по времени, число полос спектра, число точек векторскопа |

---

## Фаза 12 — A/B с отображением LUFS (P12) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| LUFS у переключателя A/B | ✅ | Под кнопками A и B отображаются значения LUFS (оригинал / мастер) после мастеринга |
| Сброс при смене файла/очистке | ✅ | При загрузке нового файла, сбросе или старте нового мастеринга значения сбрасываются в «—» |

---

## Фаза 11 — Мобильная адаптация (P11) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| viewport + viewport-fit=cover | ✅ | Поддержка safe-area (вырезы/островки) |
| Breakpoint 640px | ✅ | Сетка жанров 2 колонки, блок цепочки с wrap, отступы .page, result-grid в колонку |
| Breakpoint 480px | ✅ | Жанры в 1 колонку, цепочка/пресеты в колонку, тосты на всю ширину + safe-area-bottom |
| Breakpoint 360px | ✅ | Уменьшенные шрифты лого и заголовка |
| Touch targets (pointer: coarse) | ✅ | min-height 44px у кнопки мастеринга, карточек жанров, заголовка цепочки; 36px у кнопок пресетов и A/B |
| -webkit-tap-highlight-color | ✅ | Убрана подсветка тапа на основных кнопках и карточках |

---

## Фаза 10 — Сохранение пресетов цепочки (P10) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Модель SavedPreset в database.py | ✅ | user_id, name, config (JSON), style, target_lufs, created_at |
| GET /api/auth/presets | ✅ | Список сохранённых пресетов пользователя |
| POST /api/auth/presets | ✅ | Сохранить пресет (name, config, style, target_lufs) |
| GET /api/auth/presets/{id} | ✅ | Получить один пресет для загрузки в цепочку |
| DELETE /api/auth/presets/{id} | ✅ | Удалить пресет |
| UI: блок в карточке «Цепочка модулей» | ✅ | Поле имени + «Сохранить пресет», выбор из списка + «Загрузить» / «Удалить» (видно только залогиненным) |

---

## Фаза 9 — Пакетная обработка (Batch) ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| POST /api/v2/batch | ✅ | До 10 файлов, общие style/format/config, ответ { jobs: [{ job_id, filename }] } |
| Rate limit для batch | ✅ | Каждый файл = 1 использование; Free: 429 если remaining < кол-ва файлов |
| Фоновое выполнение | ✅ | По одному background task на файл, клиент опрашивает /api/master/status/{id} |
| UI: multiple file input | ✅ | Секция «Пакетная обработка» (Pro/debug), «Выбрать файлы» + список имён |
| UI: список задач batch | ✅ | Панель с прогрессом по каждому файлу и ссылкой «Скачать» по готовности |

---

## Фаза 8 — Режим отладки ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| MAGIC_MASTER_DEBUG в config | ✅ | `config.py`: debug_mode (bool), парсинг 1/true/yes |
| GET /api/debug-mode | ✅ | Ответ `{ "debug": true }` при включённом флаге — для фронта |
| /api/limits в debug | ✅ | Без авторизации возвращать tier=pro, limit=-1, remaining=-1 |
| Rate limit bypass в debug | ✅ | POST /api/master и /api/v2/master не проверяют лимит и не вызывают _record_usage |
| UI: бейдж «Режим отладки» | ✅ | Жёлтый бейдж «Режим отладки · все функции без входа», разблокированы все карточки и MP3/FLAC |
| README | ✅ | Раздел «Режим отладки» с примером `MAGIC_MASTER_DEBUG=1 bash start.sh` |

---

## Фаза 5 — Портал + тарифные ограничения 🔄

| Задача | Статус | Описание |
|--------|--------|----------|
| Лендинг-страница (landing.html) | ✅ | Hero + drag-drop, How it works, Features, Pricing 3 тарифа, FAQ, Footer |
| Маршрутизация `/`, `/app`, `/pricing` | ✅ | FastAPI FileResponse для landing/app/pricing |
| Передача файла с лендинга в приложение | ✅ | sessionStorage base64 + автозагрузка в app.js |
| Навигация «← На главную» в приложении | ✅ | Ссылка в header |
| Rate limiting Free tier (3/день по IP) | ✅ | `_check_rate_limit`, `_record_usage`; HTTP 429 при превышении |
| GET /api/limits — текущий лимит и остаток | ✅ | Ответ: tier, used, limit, remaining, reset_at |
| Tier indicator в приложении | ✅ | Бейдж «Free · N осталось» с цветом (зелёный/жёлтый/красный) |
| Locked controls для Free (жанры, MP3/FLAC) | ✅ | `data-tier="pro"` + CSS `.locked` + иконка 🔒 на карточках |
| Upgrade modal при попытке использовать Pro-функцию | ✅ | Попап с описанием, списком фич и ссылкой на /pricing |
| pricing.html — отдельная страница тарифов | ✅ | Free/Pro/Studio с ценами, таблицей сравнения, FAQ, toggle ежемес/ежегод |
| Тесты rate limiting | ✅ | test_api_limits_*, test_api_v2_master_rate_limit (HTTP 429) |

---

## Фаза 6 — Аутентификация 🔄

| Задача | Статус | Описание |
|--------|--------|----------|
| SQLite база + модель User | ✅ | `database.py`: SQLAlchemy, таблица users (id, email, hashed_password, tier, created_at) |
| Утилиты JWT + bcrypt | ✅ | `auth.py`: create_access_token, verify_token, get_password_hash, verify_password |
| POST /api/auth/register | ✅ | Регистрация email+password → JWT токен; tier="pro" в бета-период |
| POST /api/auth/login | ✅ | Логин email+password → JWT токен |
| GET /api/auth/me | ✅ | Профиль по Bearer токену: email, tier, user_id |
| POST /api/auth/logout | ✅ | Endpoint для единообразия (клиент удаляет токен) |
| Зарегистрированные = Pro тариф | ✅ | `/api/limits` возвращает tier=pro, limit=-1 для авторизованных |
| Rate limit bypass для авторизованных | ✅ | `/api/v2/master` и `/api/master` пропускают check_rate_limit если есть Bearer |
| frontend/login.html | ✅ | Форма входа, валидация, редирект после логина |
| frontend/register.html | ✅ | Форма регистрации, индикатор силы пароля, перки Pro |
| Шапка index.html — auth state | ✅ | «email + Pro + Выйти» или «Free badge + Войти» |
| Tier badge — Pro для залогиненных | ✅ | Pro-бейдж + разблокировка карточек + MP3/FLAC после входа |
| Auth заголовок в fetch-запросах | ✅ | `authHeaders()` передаётся в /api/v2/master и /api/v2/analyze |

---

---

## Фаза 7 — История мастерингов + Dashboard 🔄

| Задача | Статус | Описание |
|--------|--------|----------|
| MasteringRecord в БД | ✅ | `database.py`: таблица mastering_records (user_id FK, filename, before_lufs, after_lufs, style, format, duration_sec, created_at) |
| POST /api/auth/record — сохранить запись | ✅ | Фронт отправляет метаданные после скачивания файла |
| GET /api/auth/history — история + stats | ✅ | last 30 записей + stats: total, avg_lufs_change, top_style |
| DELETE /api/auth/history/{id} | ✅ | Удалить запись из истории |
| POST /api/auth/change-password | ✅ | Смена пароля с проверкой старого |
| frontend/dashboard.html | ✅ | Профиль+аватар, 4 stat-карточки, таблица истории, смена пароля |
| Маршрут /dashboard | ✅ | FileResponse в main.py |
| Ссылка «Дашборд» в шапке index.html | ✅ | Рядом с email в auth row |
| app.js — отправка записи в API | ✅ | После скачивания файла POST /api/auth/record (silent, если залогинен) |

---

## Фаза 0 — Текущий спринт ✅

| Задача | Статус |
|--------|--------|
| fix-ffprobe-pipeline | ✅ |
| fix-ffprobe-errors | ✅ |
| fix-frontend-toast | ✅ |
| daw-css | ✅ |
| daw-html | ✅ |
| daw-js-core | ✅ |
| daw-js-integrate | ✅ |

---

## Фаза 1 — Аудит-фиксы + Модульный рефакторинг ✅

### 1.1 Аудит-фиксы ✅

| Задача | Статус |
|--------|--------|
| config-prefix (MAGIC_MASTER_) | ✅ |
| config-default-lufs в API | ✅ |
| GET /api/health | ✅ |
| jobs-cleanup (TTL, max_entries) | ✅ |
| deps-cleanup (pedalboard удалён) | ✅ |

### 1.2 Модульный рефакторинг ✅

| Задача | Статус |
|--------|--------|
| BaseModule, MasteringChain, ms_utils.py | ✅ |
| Модули: dc_offset, peak_guard, equalizer, dynamics, normalize_lufs, exciter, imaging, maximizer | ✅ |
| POST /api/v2/master | ✅ |
| Maximizer lookahead (6 ms) | ✅ |
| transient_aware maximizer | ✅ (algorithm + sensitivity в цепочке) |

---

## Фаза 2 — Расширение процессоров 🔄

| Задача | Статус |
|--------|--------|
| EQ: linear_phase FFT-based + M/S per-band | ✅ linear_phase для target_curve (phase_mode в конфиге; FFT overlap-add, та же АЧХ). M/S per-band — в плане |
| Dynamics: variable crossovers, soft knee, upward compression | ✅ soft knee, crossovers, upward (band_ratios: ratio < 1) |
| Imager: 4-band, stereoize, correlation meter | ✅ stereoize, 4-band, correlation meter (POST /api/v2/analyze → stereo_correlation, отображение в карточке замера) |
| Exciter: tape/tube/transistor/warm/digital modes + oversampling 1×/2×/4× | ✅ |
| Maximizer: transient_aware алгоритм | ✅ (algorithm + sensitivity в цепочке) |

---

## Фаза 3 — Новые модули 🔄

| Задача | Статус |
|--------|--------|
| Reverb: 5 типов (plate, room, hall, theater, cathedral) | ✅ apply_reverb (Schroeder comb+allpass), ReverbModule, в цепочке (enabled: false по умолчанию) |
| Dithering: noise-shaped, auto-blank | ✅ export_audio: dither_type (tpdf, ns_e), auto_blank_sec; _write_wav_16bit_dithered(dither_type) |
| **POST /api/v2/analyze (базовый)** | ✅ lufs, peak_dbfs, duration_sec, sample_rate, channels, stereo_correlation |
| Расширение analyze: FFT, LUFS timeline, vectorscope | ✅ spectrum_bars, lufs_timeline, vectorscope_points (стерео, до 1000 точек) при extended=1 |

---

## Фаза 4 — Frontend v2 ✅

| Задача | Статус |
|--------|--------|
| Вынести JS в app.js | ✅ Подключён в index.html |
| Вынести CSS в styles.css | ✅ В index.html добавлен `<link rel="stylesheet" href="styles.css">`; скрипт extract_assets.py (авто-поиск границ). Запуск: `python3 frontend/extract_assets.py` из корня — создаёт styles.css; инлайн-стили остаются как fallback. |
| Модульные карточки (per-module Amount) | ✅ Слайдер Amount в карточке цепочки, blend в BaseModule |
| Phase mode в цепочке (target_curve) | ✅ Селект «Min. фаза» / «Linear phase» в карточке «Цепочка модулей», отправка в POST /api/v2/master |
| Reverb type в цепочке (reverb) | ✅ Селект Plate/Room/Hall/Theater/Cathedral в карточке «Цепочка модулей», отправка в POST /api/v2/master |
| Корреляция L/R в замере | ✅ Поле stereo_correlation в POST /api/v2/analyze, отображение «Корреляция L/R» под VU-метром |
| График LUFS по времени | ✅ При extended=true приходит lufs_timeline; отрисовка под метром (canvas, заливка под кривой) |
| Спектроанализатор (canvas, лог. шкала 20 Hz–20 kHz) | ✅ FFT 4096, 64 полосы, показ при загрузке файла |
| Векторскоп (canvas, Lissajous L vs R) | ✅ Показ при загрузке, до 6k точек |
| Drag-and-drop порядок модулей | ✅ Перетаскивание в карточке «Цепочка модулей», отправка config в POST /api/v2/master |

---

## Итог

- **Выполнено:** Фаза 0–17: цепочка модулей, analyze, портал, тарифы, аутентификация, история, режим отладки, пакетная обработка, сохранение пресетов (P10), мобильная адаптация (P11), A/B с LUFS (P12), экспорт отчёта TXT (P13), приоритетная очередь (P14), экспорт отчёта в JSON (P15), AI-агенты (P16), **AI отчёт в UI (P17)**.
- **Осталось разработать:** см. блок «Что осталось» ниже и документ **[Функции и уровни](doc/ФУНКЦИИ_И_УРОВНИ.md)** (что можно добавить, бесплатно vs подписка).
- **Версионность:** версия в `backend/app/version.py`, отображение в API и в футере интерфейса. История изменений — [CHANGELOG.md](CHANGELOG.md).
- **Аудит:** полный аудит кода и дизайна (2026-02-28) — [doc/AUDIT_FULL_2026.md](doc/AUDIT_FULL_2026.md); ранее — [AUDIT.md](AUDIT.md).
---

## Что осталось разработать

| # | Задача | Приоритет | Описание |
|---|--------|-----------|----------|
| 1 | LUFS timeline в analyze | ✅ сделано | Краткосрочный LUFS по времени: `lufs_timeline`, `timeline_step_sec` при `extended=true`. |
| 2 | Vectorscope-данные в analyze | ✅ сделано | При extended=true и стерео в ответе vectorscope_points (до 1000 точек [l, r]); compute_vectorscope_points в pipeline. |
| 3 | M/S per-band для EQ | ✅ сделано | apply_target_curve(eq_ms=True) и TargetCurveModule(eq_ms); при стерео — кривая отдельно к Mid и Side. В цепочке: чекбокс M/S у студийного EQ. |
| 4 | Oversampling для Exciter | ✅ сделано | apply_harmonic_exciter(oversample=1|2|4), ExciterModule + default_config; в цепочке — селект 1×/2×/4× в карточке эксайтера. |
| 5 | Reverb M/S | ✅ сделано | mix_mid, mix_side в apply_reverb и ReverbModule; при стерео — реверб по M/S с разным mix. В цепочке: поля M/S (Mid/Side mix 0–100%). |
| 6 | Dither/export в API | ✅ сделано | В POST /api/v2/master добавлены form-поля dither_type (tpdf/ns_e), auto_blank_sec; передаются в export_audio. В карточке «Параметры» — блок «Экспорт (WAV/16-bit)»: выбор дизеринга и обрезки тишины (0 / 0.5 / 1 сек). |
| 7 | Вынос инлайн-стилей | ✅ сделано | extract_assets.py --no-inline: после извлечения CSS удаляет блок `<style>...</style>` из index.html. Запуск: `python3 frontend/extract_assets.py --no-inline`. |
| 8 | ITU/E-weighted dither | ✅ сделано | Добавлен ns_itu в export_audio (_dither_noise_ns_itu), выбор в UI «Экспорт (WAV/16-bit)». |

---

## План по шагам (текущее выполнение)

| Шаг | Задача | Статус |
|-----|--------|--------|
| 1 | Добавить план по шагам в PROGRESS.md | ✅ |
| 2 | Реализовать `compute_lufs_timeline()` в pipeline.py (блоки 400 ms, до 300 точек) | ✅ |
| 3 | В `POST /api/v2/analyze` при `extended=true` возвращать `lufs_timeline` и `timeline_step_sec` | ✅ |
| 4 | Обновить PROGRESS/ROADMAP — отметить LUFS timeline выполненным | ✅ |
| 5 | (Опционально) График LUFS по времени на фронте при расширенном замере | ✅ График под VU-метром; кнопка «Измерить громкость» отправляет extended=true, отображается lufs_timeline |

---

## Сколько осталось и когда отлаживать

**Итого:** все запланированные задачи выполнены. M/S per-band для EQ реализован (eq_ms в target_curve, чекбокс M/S в цепочке).

**Что ещё можно сделать:** экспорт отчёта в PDF, дополнительные форматы экспорта. Разделение функций на «без регистрации» и «по подписке» описано в [doc/ФУНКЦИИ_И_УРОВНИ.md](doc/ФУНКЦИИ_И_УРОВНИ.md).

---

## План доработок с прогрессом

### Выполнено (фазы P0–P16)

| Фаза | Название | Прогресс |
|------|----------|----------|
| P0 | Спринт DAW UI, багфиксы | 7/7 ✅ |
| P1 | Аудит-фиксы | 5/5 ✅ |
| P2 | Модульный бэкенд v2 | 4/4 + API v2 ✅ |
| P3 | Расширение процессоров (EQ, M/S, Reverb, Dither, oversampling) | ✅ |
| P4 | Frontend v2 (спектр, векторскоп, цепочка, drag-and-drop) | ✅ |
| P5 | Портал + тарифы (landing, pricing, rate limit, /api/limits) | ✅ |
| P6 | Аутентификация (SQLite, JWT, register/login) | ✅ |
| P7 | История + Dashboard | ✅ |
| P8 | Режим отладки (MAGIC_MASTER_DEBUG) | ✅ |
| P9 | Пакетная обработка (batch) | ✅ |
| P10 | Сохранение пресетов цепочки | ✅ |
| P11 | Мобильная адаптация | ✅ |
| P12 | A/B с отображением LUFS | ✅ |
| P13 | Экспорт отчёта анализа (TXT) | ✅ |
| P14 | Приоритетная очередь для Pro | ✅ |
| P15 | Экспорт отчёта в JSON | ✅ |
| P16 | AI-агенты (рекомендатор, отчёт, авто-мастеринг, NL→config, чат), OpenAI + DeepSeek, UI кнопки | ✅ |

### Предлагаемые следующие шаги (кандидаты на P17+)

| # | Задача | Приоритет | Описание |
|---|--------|-----------|----------|
| 1 | **AI отчёт в UI** | ✅ P17 | Кнопка «AI отчёт» после замера: вызов POST /api/ai/report, показ summary + recommendations под блоком отчёта. |
| 2 | **Чат-помощник в UI** | ✅ P24 | Плавающая кнопка-чат + панель с историей, POST /api/ai/chat с контекстом анализа. |
| 3 | **NL→настройки в UI** | ✅ P25 | Поле «Опишите желаемые изменения» в цепочке, POST /api/ai/nl-config → подстановка в цепочку и LUFS. |
| 4 | **Экспорт отчёта в PDF** | ✅ P27 | Кнопка «PDF» рядом с TXT/JSON; window.open + print, включает AI-отчёт если открыт. |
| 5 | **Доп. форматы экспорта** | ✅ P30 | OPUS 192 kbps (libopus/ffmpeg); pipeline.py + UI-опция в селекте. |
| 6 | Тесты для AI API | ✅ P26 | pytest для /api/admin/*, /api/payments/* (17 тестов: CRUD users, news, campaigns, transactions, webhook). |
| 7 | **Авто-проверка подписки** | ✅ P28 | check_and_expire_subscription в database.py; _get_current_user_optional понижает тариф если срок вышел. |
| 8 | **Вкладка Настройки в Admin** | ✅ P29 | GET /api/admin/settings; sidebar + страница Настройки с разделами: App, SMTP, YooKassa. |

| 9  | **Профиль пользователя** | ✅ P31 | /profile: тариф, статус подписки, история мастерингов, смена пароля; GET /api/auth/profile. |
| 10 | **Email: истечение подписки** | ✅ P32 | send_subscription_expiry_warning_email (за 3 дня) + send_subscription_expired_email (после даунгрейда). |
| 11 | **Brute-force защита Auth** | ✅ P33 | _check_auth_rate_limit: 10 попыток/мин с IP на /api/auth/login и /api/auth/register. |

| 12 | **Сброс пароля по email** | ✅ P35 | POST /api/auth/forgot-password + /reset-password; forgot-password.html, reset-password.html; ссылка в login.html. |
| 13 | **Безопасные миграции БД** | ✅ P36 | _run_migrations(): PRAGMA table_info + ALTER TABLE ADD COLUMN для новых колонок без сноса данных. |
| 14 | **SSE прогресс мастеринга** | ✅ P37 | GET /api/master/progress/{id} StreamingResponse; waitForJobCompletion() с fallback на polling; X-Accel-Buffering:no для Nginx. |

| 15 | **pytest Auth** | ✅ P38 | test_auth.py: 26 тестов — register/login, rate limit P33, профиль P31, change-password P34, forgot/reset P35, history. |
| 16 | **Admin CSV-экспорт** | ✅ P39 | GET /api/admin/users/export.csv + /transactions/export.csv; UTF-8 BOM для Excel; кнопки ⬇ CSV в admin.html. |
| 17 | **Docker Production** | ✅ P40 | docker-compose.yml (dev) + docker-compose.prod.yml (app+nginx); .env.example со всеми переменными; nginx SSE-настройки. |

| 18 | **Email верификация** | ✅ P41 | MAGIC_MASTER_REQUIRE_EMAIL_VERIFY; is_verified в User; /api/auth/verify-email + resend; verify-email.html. |
| 19 | **User History CSV** | ✅ P42 | GET /api/auth/history/export.csv; кнопка «⬇ CSV» в profile.html. |
| 20 | **Admin Dashboard Analytics** | ✅ P43 | SVG sparklines (7 дней): пользователи/мастеринги/выручка; tierBar; активные подписки. |
| 21 | **CHANGELOG.md** | ✅ P44 | Полная документация всех фаз P0–P44 в CHANGELOG.md. |

| 22 | **A/B аудио плеер** | ✅ P45 | HTML5 audio + custom UI; before/after через /api/master/preview/{id}?src=original|mastered; переключение без потери позиции. |
| 23 | **Глобальный rate limit API** | ✅ P46 | middleware: 300 req/min с IP; MAGIC_MASTER_GLOBAL_RATE_LIMIT; Retry-After заголовок; исключение SSE. |
| 24 | **Страница статуса /status** | ✅ P47 | /api/health расширен: БД, диск, ffmpeg, jobs, uptime, python; status.html авто-обновление 30 с. |

| 25 | **Admin bulk-действия** | ✅ P48 | POST /api/admin/users/bulk-action: block/unblock/delete/set_tier; чекбоксы + Select All в таблице; bulk bar. |
| 26 | **PWA** | ✅ P49 | manifest.json + sw.js (Cache-First статика, Network-First API); SW-регистрация; кнопка «Установить»; meta-теги Apple. |
| 27 | **Admin DB Backup** | ✅ P50 | GET /api/admin/backup/db: VACUUM INTO горячий бэкап; кнопка «⬇ Backup DB» в настройках; DATABASE_URL в database.py. |

| 28 | **Telegram-уведомления** | ✅ P51 | notifier.py: notify_new_user, notify_payment, notify_mastering_error, notify_server_startup; тест из Admin. |
| 29 | **API-ключи для Pro/Studio** | ✅ P52 | ApiKey модель; create/list/revoke /api/auth/api-keys; X-API-Key заголовок; раздел в profile.html. |
| 30 | **Admin: тест SMTP + Telegram** | ✅ P53 | POST /api/admin/notifications/test-email + test-telegram; кнопки в секции «Настройки». |

Текущий прогресс: **фазы P0–P56 закрыты.** P56: CORS и безопасность webhook.

---

## Фаза P56 — CORS и безопасность webhook YooKassa ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| Настраиваемый CORS | ✅ | config.cors_origins (MAGIC_MASTER_CORS_ORIGINS); пусто = ["*"]; в проде задать список доменов |
| IP whitelist webhook | ✅ | config.yookassa_webhook_ip_whitelist; при заданном списке запросы только с этих IP |
| Исправление notify_payment в webhook | ✅ | В webhook передаются amount_val и currency_val вместо несуществующих amount/currency |
| Тест webhook 403 при whitelist | ✅ | test_admin.py: test_webhook_403_when_ip_not_in_whitelist |

---

## Фаза P55 — Spectral Denoiser: пресеты и опциональный порог ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| DENOISE_PRESETS в pipeline | ✅ | light (0.25, 20%), medium (0.5, 15%), aggressive (0.75, 10%); apply_spectral_denoise уже поддерживал noise_percentile |
| API denoise_preset / denoise_noise_percentile | ✅ | Form-параметры в POST /api/v2/master; pro_params в _run_mastering_job_v2 |
| UI пресет Denoiser | ✅ | Селект «Пресет»: Свой / Лёгкий / Средний / Агрессивный в карточке Spectral Denoiser |
| Тест test_denoise_presets | ✅ | test_pipeline.py: проверка DENOISE_PRESETS и apply_spectral_denoise |

---

## Фаза P54 — pytest для pipeline и AI ✅

| Задача | Статус | Описание |
|--------|--------|----------|
| test_pipeline.py | ✅ | remove_dc_offset, measure_lufs, compute_spectrum_bars, compute_vectorscope_points, compute_lufs_timeline, measure_stereo_correlation, export_audio (wav), run_mastering_pipeline, load_audio_from_bytes, STYLE_CONFIGS |
| test_ai.py | ✅ | get_ai_limit_for_tier, check_ai_rate_limit, record_ai_usage, recommend_preset (rule-based), report_with_recommendations, nl_to_config (без API), VALID_STYLES |

Запуск: `cd backend && PYTHONPATH=. python3 -m pytest tests/test_pipeline.py tests/test_ai.py -v`

---

## Бэклог (P60+, не в текущем спринте)

План P0–P59 выполнен. Ниже — идеи для будущих итераций (подключать по запросу).

| # | Задача | Описание |
|---|--------|----------|
| 1 | ~~Доп. форматы экспорта~~ | ✅ P57: AAC (M4A 192 kbps) реализован. Остальное: другие битрейты MP3/OPUS — по запросу. |
| 2 | ~~Мониторинг и алерты~~ | ✅ P58: GET /api/metrics для скрапинга; /api/health уже есть. Доп. алерты — по запросу. |
| 3 | ~~Локализация~~ | ✅ P59: базовая i18n — ru/en, переключатель, /api/locale, /locales/*.json, data-i18n. Расширение строк — по запросу. |
| 4 | ~~Плагины/пресеты от сообщества~~ | ✅ P64: GET /api/presets/community, JSON-файл пресетов, UI в селекте. Расширение — по запросу. |
| 5 | Рефакторинг: ~~main.py~~, ~~a11y~~, ~~magic bytes~~, ~~E2E-тест~~ | По [doc/AUDIT_FULL_2026.md](doc/AUDIT_FULL_2026.md). Все подпункты ✅ P60–P63. |

---

**Когда эффективно запускать отладку**

- **Сейчас** — удачный момент: все ключевые сценарии уже есть (загрузка файла, замер LUFS/корреляции/спектра/графика LUFS, мастеринг v2 с цепочкой, экспорт WAV/MP3/FLAC с дизерингом и auto_blank). Отладка на этом этапе позволит:
  1. Проверить полный путь: загрузка → «Измерить громкость» → «Запустить мастеринг» → скачивание.
  2. Убедиться, что конфиг цепочки (phase_mode, reverb_type, mix_mid/mix_side, oversample, dither_type, auto_blank_sec) доходит до бэкенда и применяется.
  3. Поймать баги при изменении конфига цепочки (eq_ms, phase_mode и т.д.).

- **Как отлаживать:** запустить `bash start.sh`, открыть http://localhost:8000, пройти сценарии выше; при ошибках смотреть логи сервера и ответы API в DevTools (Network). Для отладки бэкенда — тесты в `backend/` или запуск с отладчиком (например, `python -m debugpy --listen 5678 -m uvicorn app.main:app`).

---

## Статус задач (сводка)

| Задача | Статус |
|--------|--------|
| LUFS timeline в analyze | ✅ |
| Vectorscope в analyze | ✅ |
| Dither/export в API + UI | ✅ |
| Oversampling Exciter + UI | ✅ |
| Reverb M/S (mix_mid, mix_side) + UI | ✅ |
| ITU/E-weighted dither (ns_itu) + UI | ✅ |
| Вынос инлайн-стилей (extract_assets.py --no-inline) | ✅ |
| M/S per-band для EQ (eq_ms в target_curve + UI) | ✅ |
| Лендинг-страница портала (landing.html) | ✅ |
| Маршрутизация `/`, `/app`, `/pricing` | ✅ |
| sessionStorage — передача файла с лендинга | ✅ |
| Навигация «На главную» в приложении | ✅ |
| Rate limiting Free (3/день по IP, HTTP 429) | ✅ |
| GET /api/limits — тариф, остаток, reset_at | ✅ |
| Tier badge в UI (Free · N осталось) | ✅ |
| Locked Pro-карточки с иконкой 🔒 | ✅ |
| Upgrade modal при клике на Pro-функцию | ✅ |
| pricing.html — полная страница тарифов | ✅ |
| Тесты rate limiting (pytest) | ✅ |
| SQLite + SQLAlchemy (database.py) | ✅ |
| JWT + bcrypt (auth.py) | ✅ |
| POST /api/auth/register + login + me | ✅ |
| frontend/register.html + login.html | ✅ |
| Auth state в шапке приложения | ✅ |
| Pro tier для залогиненных (bypass rate limit) | ✅ |
