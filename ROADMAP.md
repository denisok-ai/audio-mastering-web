# Magic Master — Дорожная карта разработки

> Документ сохранён: 2026-02-27  
> Статус проекта: **активная разработка**  
> План переработан с учётом полного аудита и приоритетов (см. [AUDIT.md](AUDIT.md)).

---

## Что уже реализовано (v1, текущее состояние)

### Бэкенд (`backend/app/`)

| Модуль | Файл | Описание |
|---|---|---|
| FastAPI-приложение | `main.py` | REST API: /api/measure, /api/master, /api/styles, /api/presets |
| Конвейер мастеринга | `pipeline.py` | DC remove → EQ → 4-band dynamics → maximizer → LUFS normalize → exciter → imager |
| Конфигурация | `config.py` | Настройки сервера, лимиты загрузки |

### Реализованные процессоры

| Процессор | Функция | Параметры |
|---|---|---|
| DC offset removal | `remove_dc_offset()` | — |
| Peak protection | `remove_intersample_peaks()` | headroom_db |
| Studio EQ | `apply_target_curve()` | Sony/Warner кривая, HP 40Hz, LP 18kHz |
| 4-band dynamics | `apply_multiband_dynamics()` | Кроссоверы 214/2230/10000 Hz, ratio, threshold, gain |
| Maximizer | `apply_maximizer()` | threshold -2.5 dB, ceiling -0.3 dB |
| LUFS normalization | `normalize_lufs()` | target_lufs, защита ±20 dB |
| Final EQ | `apply_final_spectral_balance()` | Дип 3kHz, дип 16kHz, буст низов, буст 8kHz |
| Style EQ | `apply_style_eq()` | 5 полос по жанру |
| Harmonic Exciter | `apply_harmonic_exciter()` | exciter_db, tanh 2+3 гармоники, HP > 3kHz |
| Stereo Imager | `apply_stereo_imager()` | width (Mid-Side) |

### Жанровые пресеты (`STYLE_CONFIGS`)

| Пресет | LUFS | exciter_db | imager_width | Описание |
|---|---|---|---|---|
| standard | -14.0 | 0.0 | 1.0 | Стриминг (Spotify/YouTube) |
| edm | -9.0 | 0.6 | 1.25 | EDM, электронная музыка |
| hiphop | -13.0 | 0.3 | 1.1 | Hip-hop |
| classical | -18.0 | 0.0 | 1.05 | Классика |
| podcast | -16.0 | 0.0 | 1.0 | Подкасты |
| lofi | -18.0 | 0.2 | 0.9 | Lo-fi |
| **house_basic** | **-10.0** | **0.8** | **1.3** | **House Basic (Ozone 5-inspired)** |

### Фронтенд (`frontend/index.html`)

- Drag & drop загрузка WAV/MP3/FLAC
- Waveform player с A/B сравнением оригинал/мастер
- VU-метр (LUFS, 28 сегментов)
- Жанровые карточки (7 пресетов включая House Basic)
- Progress pipeline с именованными шагами
- Before/After LUFS панель
- История мастеринга (localStorage, 8 записей)
- Toast-уведомления
- Шаги Ozone 5 Exciter / Imager (видны при house_basic)
- Адаптивный дизайн (mobile 480px)

### DLL-анализ (папка `/doc`)

Проанализированы модули iZotope Ozone 5:

| DLL | Модуль | Реализовано |
|---|---|---|
| `iZotope Ozone 5 Equalizer.dll` | Параметрический EQ | apply_target_curve + apply_style_eq |
| `iZotope Ozone 5 Dynamics.dll` | Multiband compressor | apply_multiband_dynamics |
| `iZotope Ozone 5 Maximizer.dll` | Maximizer | apply_maximizer |
| `iZotope Ozone 5 Exciter.dll` | Harmonic Exciter | apply_harmonic_exciter |
| `iZotope Ozone 5 Imager.dll` | Stereo Imager | apply_stereo_imager |
| `iZotope Insight.dll` | Metering | measure_lufs |

---

## Архитектура (текущая и целевая)

Подробный аудит: **[AUDIT.md](AUDIT.md)**.

### Текущая

- **Backend:** один процесс FastAPI; конвейер мастеринга в одном файле `pipeline.py`; задачи в памяти (`_jobs`).
- **Frontend:** одна страница `frontend/index.html` (HTML + CSS + JS).
- **Конфиг:** переменные окружения с двумя префиксами (`MASTERFLOW_` в config, `MAGIC_MASTER_` в run/deploy).

### Целевая (после изменений по аудиту и ROADMAP v2)

- **Backend:** модульный конвейер в `backend/app/modules/` (BaseModule, EQ, Dynamics, Maximizer, Exciter, Imager и т.д.) и `MasteringChain`; единый префикс настроек; очистка старых задач в памяти; health endpoint.
- **Frontend:** разделение на `index.html` + `styles.css` + `app.js`; при росте — опционально сборка (Vite) и компонентная структура.
- **Зависимости:** без неиспользуемых пакетов (pedalboard либо удалить, либо задействовать).

---

## Приоритеты разработки

| Приоритет | Блок | Задачи |
|-----------|------|--------|
| **P0** | Завершить текущий спринт | DAW Comparison UI (daw-css, daw-html, daw-js-core, daw-js-integrate) |
| **P1** | Аудит-фиксы (архитектура/операция) | Унификация env (MAGIC_MASTER_), health endpoint, очистка/лимит _jobs, использование default_target_lufs в API, удаление неиспользуемых зависимостей |
| **P2** | Модульный бэкенд v2 | BaseModule, MasteringChain, перенос pipeline в модули, POST /api/v2/master |
| **P3** | Расширение процессоров | Алгоритмы maximizer, EQ linear phase, variable crossovers, exciter/imager режимы (см. фазы 2–3 ниже) |
| **P4** | Frontend v2 | Вынести CSS/JS в файлы; спектроанализатор, векторскоп, drag-and-drop порядок модулей |

---

## Текущий спринт — задачи

### Sprint 1: Bugfix ffprobe + DAW Comparison UI

#### Выполнено

- [x] **fix-ffprobe-pipeline** — `pipeline.py`: WAV/FLAC читаются через `soundfile` без ffmpeg; FLAC экспортируется через `soundfile.write()` 24-bit PCM
- [x] **fix-ffprobe-errors** — `pipeline.py` + `main.py`: `_raise_ffmpeg_error()` с типом `NoReturn`; ранняя проверка `shutil.which("ffmpeg")` для MP3 в API
- [x] **fix-frontend-toast** — `index.html`: `friendlyError()` показывает toast с инструкцией `sudo apt-get install -y ffmpeg` при ошибках ffprobe

#### Ожидает выполнения

- [x] **daw-css** — CSS для DAW-карточки сравнения форм волн (MusicTech тёмная тема)
- [x] **daw-html** — HTML разметка `#dawCard` с ruler, двумя track-канвасами, stats-row
- [x] **daw-js-core** — JS функции: `dawComputeStats`, `dawDrawRuler`, `dawDrawWaveform`, `showDawComparison`
- [x] **daw-js-integrate** — Интеграция `showDawComparison` после мастеринга, сброс в `setFile`/`resetAll`, ResizeObserver

---

## Дорожная карта v2 — Magic Master следующего поколения

> Вдохновлён архитектурой iZotope Ozone 5.0.3

### Фаза 0 — Текущий спринт (багфиксы + DAW UI)

Описание выше в разделе "Текущий спринт". **Приоритет P0.**

---

### Фаза 1 — Аудит-фиксы + Модульный рефакторинг бэкенда

**1.1 Аудит-фиксы (приоритет P1, выполнить в начале фазы)**

- [x] **config-prefix** — Унификация переменных окружения: в `config.py` префикс `MAGIC_MASTER_`, обновлены README/DEPLOY.
- [x] **config-default-lufs** — В `main.py` при отсутствии `target_lufs` в форме используется `settings.default_target_lufs`.
- [x] **api-health** — Добавлен `GET /api/health` с ответом `{"status": "ok"}`.
- [x] **jobs-cleanup** — Очистка `_jobs`: TTL для done/error (`MAGIC_MASTER_JOBS_DONE_TTL_SECONDS`), лимит записей (`MAGIC_MASTER_JOBS_MAX_ENTRIES`), вызов `_prune_jobs()` при создании задачи и при запросе статуса.
- [x] **deps-cleanup** — Удалён pedalboard из requirements.txt, в pipeline.py исправлен комментарий зависимостей (pydub вместо librosa).

**1.2 Модульный рефакторинг (приоритет P2)**

**Новые файлы:**
```
backend/app/
├── modules/
│   ├── base.py       ← BaseModule(ABC): process(), enabled, amount: float, ms_mode
│   ├── maximizer.py  ← 5 алгоритмов лимитинга
│   ├── equalizer.py  ← Pre/Post EQ, linear/min/mixed phase
│   ├── dynamics.py   ← 4-band, variable crossovers, upward comp
│   ├── imaging.py    ← 4-band M/S, stereoize, correlation
│   ├── exciter.py    ← 4 saturation modes, oversampling
│   ├── reverb.py     ← FDN 8×8, 5 types
│   ├── dithering.py  ← TPDF/noise-shaped, auto-blank
│   └── metering.py   ← FFT, LUFS, vectorscope
├── chain.py          ← MasteringChain: from_config(), process()
└── ms_utils.py       ← mid_side_encode/decode
```

**Новый API:** `POST /api/v2/master` принимает JSON-конфиг цепочки модулей.

**Задачи фазы 1.2:**
- [x] Создать `BaseModule`, `MasteringChain`, `ms_utils.py` (mid_side_encode/decode)
- [x] Перенести существующие функции pipeline.py в классы-модули (v1 совместимость: `run_mastering_pipeline` без изменений)
- [x] `POST /api/v2/master` с опциональным JSON chain config; без config — `default_chain(target_lufs, style)`
- [x] Maximizer: алгоритмы `lookahead` и `transient_aware` (envelope + blend по маске), `MaximizerModule`: algorithm, sensitivity

---

### Фаза 2 — Расширение процессоров

#### Maximizer — 5 алгоритмов

| ID | Алгоритм | Характер |
|---|---|---|
| `brickwall` | Жёсткий клиппер | Агрессивный |
| `soft_knee` | Текущий (мягкое колено) | Нейтральный |
| `lookahead` | Lookahead 6ms | Прозрачный |
| `transient_aware` | Onset detector + var.gain | Сохраняет атаки (реализовано: sensitivity 0–1) |
| `irc_smooth` | Intelligent Release Control | Максимально прозрачный |

Новые параметры: `transient_recovery` (0–1), `character` (fast↔slow), `oversampling` (1/2/4×), `true_peak` (bool)

#### EQ Pre/Post — 8 полос, 3 режима фазы

| Режим | Реализация | Latency |
|---|---|---|
| `minimum_phase` | scipy IIR biquad | 0 |
| `linear_phase` | FFT overlap-add | N/2 samples |
| `mixed` | LP для >1kHz, FFT для ≤1kHz | ~N/4 |

Per-band M/S: `ms_channel` = "both" / "mid" / "side"

#### Multiband Dynamics — variable crossovers + upward compression

- Регулируемые кроссоверы (было фиксированы: 214/2230/10000 Hz)
- `knee_db` 0–40 (мягкое колено) на каждую полосу
- Upward compression: ratio < 1:1 усиливает тихие участки
- `channel_mode`: stereo / L-R / M-S
- Детекторы: rms, peak, tilt, highpass

#### Stereo Imager — 4 полосы + stereoize

- `width` per band: 0 (моно) → 1 (оригинал) → 2 (double wide)
- `stereoize`: Haas delay до 15ms (синтетическое расширение)
- Correlation meter: dot(L,R)/(‖L‖·‖R‖)

#### Harmonic Exciter — 5 режимов насыщения

| Режим | Формула | Характер |
|---|---|---|
| `transistor` | y = x - x³/3 (odd) | Чёткий |
| `tape` | y = tanh(k·x)/k | Тёплый |
| `tube` | y = x + α·x² (even) | Ламповый |
| `warm` | tape + tube mix | Аналоговый |
| `digital` | Fold-back clipping | Яркий |

Oversampling: `resample_poly` 1×/2×/4×. Параметр `delay_ms` 0–100ms.

**Задачи фазы 2:**
- [x] EQ: linear_phase для студийной кривой (target_curve: phase_mode "minimum" | "linear_phase"; FFT overlap-add). [x] M/S per-band (eq_ms: применять кривую отдельно к Mid и Side, чекбокс в цепочке).
- [x] Dynamics: soft knee, crossovers, upward (band_ratios: ratio < 1 по полосам, max_upward_boost_db)
- [x] Imager: stereoize (Haas: stereoize_delay_ms, stereoize_mix), 4-band (band_widths, crossovers_hz), correlation meter (measure_stereo_correlation в pipeline; POST /api/v2/analyze → stereo_correlation; отображение в UI замера)
- [x] Exciter: tape/tube/transistor/warm/digital modes (параметр `mode` в цепочке). [x] oversampling 1×/2×/4× (параметр `oversample` в цепочке, UI в карточке модулей).

---

### Фаза 3 — Новые модули

#### Reverb — FDN 8×8, 5 типов

| Тип | Decay | Характер |
|---|---|---|
| `plate` | 0.5–4s | Классический |
| `room` | 0.2–1.5s | Натуральный |
| `hall` | 1.5–5s | Торжественный |
| `theater` | 2–8s | Монументальный |
| `cathedral` | 5–20s | Огромный |

M/S контроль: `ms_center_dry` + `ms_side_wet`

#### Dithering — noise-shaped

| Тип | Применение |
|---|---|
| `tpdf` | Текущий стандарт |
| `rpdf` | Минимальный шум |
| `ns_e_weighted` | 16-bit для стриминга |
| `ns_itu` | Broadcast |

Параметры: `target_bits` (8/16/24), `auto_blank`, `dc_remove`

#### Metering API

- [x] **Базовый** `POST /api/v2/analyze`: загрузка файла → ответ `lufs`, `peak_dbfs`, `duration_sec`, `sample_rate`, `channels` (для спектроанализатора/векторскопа на фронте).
- [x] **Расширенный** analyze: при `extended=true` (Form) в ответе `spectrum_bars`, `lufs_timeline`, `timeline_step_sec`; для стерео — `vectorscope_points` (до 1000 точек [l, r]). correlation в базовом ответе.

**Задачи фазы 3:**
- [x] Reverb: алгоритмический (Schroeder comb+allpass), 5 типов (plate, room, hall, theater, cathedral), ReverbModule в цепочке (enabled: false по умолчанию). [x] M/S mix (mix_mid, mix_side в apply_reverb и UI в цепочке).
- [x] Dithering: TPDF + noise-shaped (ns_e) в export_audio, auto_blank_sec (обрезка тишины в конце). POST /api/v2/master принимает dither_type и auto_blank_sec; UI — выбор в карточке «Параметры». [ ] ITU/E-weighted опции в API
- [x] Расширение `/api/v2/analyze`: FFT (spectrum_bars), LUFS timeline (lufs_timeline, timeline_step_sec), vectorscope_points для стерео.

---

### Фаза 4 — Frontend v2 (приоритет P4)

- [x] Вынести JS в `app.js` — подключён в `index.html` через `<script src="app.js"></script>`
- [x] Вынести CSS в `styles.css`: в `index.html` добавлен `<link rel="stylesheet" href="styles.css">`; скрипт `extract_assets.py` извлекает CSS по тегам `<style>`/`</style>`. Запуск: `python3 frontend/extract_assets.py` — создаёт `frontend/styles.css`. Инлайн-стили — fallback. `--no-inline`: удаляет блок `<style>...</style>` из index.html после извлечения.
- [x] Карточка «Цепочка модулей» + per-module Amount (слайдер 0–100%, blend в BaseModule)
- [x] Спектроанализатор (canvas, логарифмическая шкала 20 Hz – 20 kHz, FFT 4096, 64 полосы)
- [x] Векторскоп (canvas, Lissajous L vs R, до 6k точек)
- [x] Drag-and-drop порядок модулей в цепочке (перетаскивание в карточке, отправка config в POST /api/v2/master)

---

## Пользовательские сценарии (Use Cases)

| # | Сценарий | Модули | Ключевые настройки |
|---|---|---|---|
| UC-1 | Стриминг -14 LUFS True Peak ≤ -1 | Maximizer + Dither | `irc_smooth`, `true_peak=True`, `ns_e_weighted` 16-bit |
| UC-2 | House -10 LUFS клуб | Dynamics + Exciter + Maximizer | 4-band comp, `tape` sub, `brickwall` ×4 oversample |
| UC-3 | Тональная коррекция | Pre-EQ + Post-EQ | `linear_phase`, -2dB@300Hz, +1.5dB@12kHz |
| UC-4 | Расширение стерео (моно-safe) | Imager | B2–B3 width=1.4, B1 width=0.8, correlation check |
| UC-5 | Аналоговое тепло | Exciter | `tube` 200–2kHz, `tape` 50–200Hz, drive=0.3 |
| UC-6 | Мастеринг классики | EQ + Reverb + Dynamics + Dither | `hall` mix=0.08, ratio 1.5:1, -18 LUFS |
| UC-7 | Подготовка к vinyl | Exciter + Imager + Maximizer | `warm` all bands, B1 width=0.6, ceiling=-3.0 |
| UC-8 | Подкаст/голос | Dynamics + EQ | M/S Mid 4:1, HP 80Hz, +2dB@3kHz |

---

## Порядок запуска при следующей сессии

```bash
# 1. Перейти в папку проекта
cd /home/denisok/projects/audio-mastering-web

# 2. Запустить сервер (установит зависимости если нужно)
bash start.sh

# 3. Открыть в браузере
# http://localhost:8000
```

### Рекомендуемый порядок выполнения (по приоритетам)

1. **P0 — Текущий спринт:** завершить DAW Comparison UI (daw-css → daw-html → daw-js-core → daw-js-integrate).
2. **P1 — Аудит-фиксы:** config-prefix, config-default-lufs, api-health, jobs-cleanup, deps-cleanup (см. Фаза 1.1).
3. **P2 — Модульный бэкенд:** BaseModule, MasteringChain, перенос pipeline в модули, API v2 (Фаза 1.2).
4. **P3–P4:** расширение процессоров и Frontend v2 по фазам 2–4.

### Следующий шаг разработки

**Приоритет P2 выполнен (модульный бэкенд v2).** Дальше: **P3** — расширение процессоров (Фаза 2: EQ linear phase, variable crossovers, exciter/imager режимы) или **P4** — Frontend v2 (вынос CSS/JS, спектроанализатор, векторскоп).

**Прогресс выполнения плана:** см. [PROGRESS.md](PROGRESS.md).
