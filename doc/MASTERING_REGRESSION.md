# Регрессия качества мастеринга и логи production

## 1. Расширенное логирование на сервере

В [`.env`](../.env.example) на VPS:

```env
MAGIC_MASTER_MASTERING_TRACE=1
# опционально — LUFS после каждого этапа (дороже CPU):
# MAGIC_MASTER_MASTERING_TRACE_LUFS_STAGES=1
```

Затем:

```bash
sudo systemctl restart magic-master
```

Запустите мастеринг проблемного файла (UI или `POST /api/v2/master`). Сохраните **`job_id`** из ответа.

**Снятие логов** (записи идут в **journald** под логгером `app.mastering_trace`, префикс сообщения `mastering_trace`):

```bash
sudo journalctl -u magic-master --since "15 min ago" --no-pager | grep mastering_trace
# или по job_id:
sudo journalctl -u magic-master --since "15 min ago" --no-pager | grep 'job_id=ВАШ_UUID'
```

Ожидаемые строки: `mastering_trace_start`, `mastering_trace_chain` (список модулей v2), серии `mastering_trace … stage=… peak_db=…`, `mastering_trace_done` / при ошибке `mastering_trace_error` и traceback.

После анализа **отключите** trace (`MAGIC_MASTER_MASTERING_TRACE=0` или удалите строку) и снова `systemctl restart magic-master`, чтобы не раздувать журнал.

### Отладка: снять лимиты мастеринга (временно)

Пока крутите много прогонов с `localhost` или без токенов, в `.env`:

```env
MAGIC_MASTER_MASTERING_DEBUG_SKIP_LIMITS=1
```

Это отключает недельный лимит Free по IP, **не** списывает токены Pro/Studio и **не** применяет дневной cap — удобно для поиска шума и сверки с trace. Эквивалентно по эффекту для мастеринга и `MAGIC_MASTER_DEBUG=1`, но **не** открывает debug-режим API целиком. **Снимите флаг после отладки** на публичном сервере.

Ограничение диска journald и пакет `logrotate` для nginx: [deploy/journald/README.md](../deploy/journald/README.md), общий чек-лист сервера: [PRODUCTION_DRIFT.md](PRODUCTION_DRIFT.md).

## 2. Офлайн: окна времени и стадии

Модуль [backend/app/qa/mastering_regression.py](../backend/app/qa/mastering_regression.py):

- окна по умолчанию: 2–10 с, 75–90 с, 154–160 с (под длинный трек);
- метрики по окну: `hf_rms` (ВЧ выше 8 kHz), `max_abs_diff` (макс. |Δсэмпл|), `rms`;
- `run_default_chain_stages` / `metrics_after_each_stage` — та же цепочка, что v2 Stream (`standard`, -14 LUFS), включая `v2_output_fade_in`.

## 3. Автотесты

```bash
cd backend
./venv/bin/python -m pytest tests/test_mastering_regression_windows.py -v
```

- Без WAV-фикстуры выполняется **синтетический** тест на ~48 с.
- С файлом: `test_output/_Alors On Danse Rem.wav` в корне репозитория подхватывается автоматически; иначе см. [backend/tests/fixtures/mastering_regression/README.md](../backend/tests/fixtures/mastering_regression/README.md) и **`MM_REGRESSION_WAV`**.

Опциональные пороги: скопируйте `expected_metrics.json.example` → `expected_metrics.json` рядом с WAV.

## 4. Правки DSP (пример)

Сглаживание огибающей **de-esser** и более мягкие attack/release уменьшают ВЧ-«зерно» на v1-пайплайне (`apply_deesser` в [pipeline.py](../backend/app/pipeline.py)). Цепочка v2 Stream по умолчанию **не** содержит отдельного de-esser-модуля; при необходимости смотрите стадии `dynamics`, `final_spectral_balance`, `style_eq` по логам и метрикам после каждого модуля.
