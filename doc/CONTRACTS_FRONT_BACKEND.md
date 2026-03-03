# Контракты фронт–бекенд (чек-лист п.8)

Краткая сводка по основным эндпоинтам: что шлёт фронт и что ожидает/возвращает бекенд.

## Измерить громкость

- **Фронт:** `POST /api/v2/analyze` — FormData: `file`, `extended` = `'true'`.
- **Бекенд:** `api_v2_analyze(file, extended)` в [routers/mastering.py](../backend/app/routers/mastering.py). При успехе: JSON с `lufs`, `peak_dbfs`, `duration_sec`, при `extended=true` — `spectrum_bars`, `vectorscope_points` и др.
- **Фронт читает:** `lastAnalyzeReport` (lufs, peak_dbfs, duration_sec, spectrum_bars, vectorscope_points), обновляет LUFS и спектр.

## AI отчёт

- **Фронт:** `POST /api/ai/report` — либо FormData с `file`, либо JSON с `body.analysis` (если есть `lastAnalyzeReport`).
- **Бекенд:** `api_ai_report` принимает `file: Optional[UploadFile]` и опционально тело с `analysis`. Ответ: `summary`, `recommendations`.
- **Фронт читает:** `summary`, `recommendations` для блока отчёта.

## Рекомендовать пресет (AI)

- **Фронт:** `POST /api/ai/recommend` — FormData с `file` или JSON с `analysis`.
- **Бекенд:** `api_ai_recommend`. Ответ: пресет/рекомендации.
- **Фронт читает:** данные для подстановки пресета.

## Запустить мастеринг (ручной)

- **Фронт:** `POST /api/v2/master` — FormData: `file`, `target_lufs`, `out_format`, `style`, `config` (JSON цепочки модулей), опционально `dither_type`, `auto_blank_sec`, `bitrate`, PRO-поля (`denoise_preset`, `deesser_*`, `transient_*`, `parallel_mix`, `dynamic_eq_enabled`).
- **Бекенд:** `api_master_v2` возвращает `job_id`. Статус: `GET /api/master/status/{job_id}`, результат: `GET /api/master/result/{job_id}` (бинарный файл).
- **Фронт читает:** `job_id`, опрашивает статус, при `status: done` запрашивает результат и скачивает.

## Авто-мастеринг

- **Фронт:** `POST /api/v2/master/auto` — FormData: `file`, `out_format`, опционально `bitrate`.
- **Бекенд:** `api_v2_master_auto` возвращает `job_id`; при лимите 429 — JSON `{"detail": "Лимит Free-тарифа..."}`.
- **Фронт:** при `!r.ok` показывает в тосте `detail` из ответа (через `safeResponseJson` / `friendlyError`).

## Ошибки 429 / 500

- **Бекенд:** всегда возвращает JSON с полем `detail` (строка).
- **Фронт:** при `!response.ok` парсит JSON, извлекает `detail` и показывает в тосте; при 429 пользователь видит сообщение о лимите, а не «Внутренняя ошибка сервера».

## Проверка

- Загрузка файла → «Измерить громкость» → 200, обновление LUFS и спектра.
- «AI отчёт» при наличии файла → 200, блок отчёта заполнен.
- «Запустить мастеринг» → 200 + `job_id`, опрос до `done`, скачивание.
- Исчерпание лимита (429) → ответ 429 с `detail` → тост с текстом лимита.
