# Magic Master — веб-приложение автоматического мастеринга

Веб-версия приложения для автоматического мастеринга аудио в два клика: **загрузить** → **обработать и экспортировать**.

## Возможности

- **Форматы**: WAV, MP3, FLAC (через FFmpeg / Pydub).
- **Стандарты уровня Sony/Warner**: целевой LUFS (Spotify -14, Apple -16, клубы -9), True Peak ≤ -1 dBTP, двухстадийная компрессия, студийный EQ, TPDF-дизеринг при экспорте в 16-bit.
- **Безопасность**: удаление DC-смещения, ограничение межвыборочных пиков.
- **Нормализация**: выбор целевой громкости (LUFS) и пресетов.
- **Интерфейс**: одна страница — загрузка, кнопка «Magic Master», индикатор громкости, экспорт в WAV/MP3/FLAC.

## Требования

- **Python 3.10+**
- **FFmpeg** — должен быть установлен в системе и доступен в `PATH` (для Pydub).

### Установка FFmpeg (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

### Установка venv (если `python3 -m venv` не работает)

На Ubuntu/Debian иногда нужно доустановить пакет:

```bash
sudo apt install python3.10-venv
```

## Как просто открыть по ссылке

1. Откройте **терминал** (Ctrl+Alt+T).
2. Выполните одну команду (при первом запуске запросит пароль для установки пакетов):

   `cd /home/imya_polzovatelya/Projects/Testproject/audio-mastering-web && ./start.sh`

3. В браузере откройте: **http://localhost:8000**  
   Остановка сервера: в терминале **Ctrl+C**.

---

## Локальная установка на вашей машине (пошагово)

1. **Один раз установите системные зависимости** (в терминале, потребуется пароль):

```bash
sudo apt update
sudo apt install -y ffmpeg libatomic1 python3.10-venv python3-pip
```

(Пакет **libatomic1** нужен для Pedalboard; **ffmpeg** — для загрузки и экспорта MP3/FLAC.)

2. **Установите и запустите приложение** — из корня проекта выполните:

```bash
cd /home/imya_polzovatelya/Projects/Testproject/audio-mastering-web
chmod +x install_and_run.sh
./install_and_run.sh
```

Скрипт создаст виртуальное окружение в `backend/venv`, установит зависимости и запустит сервер. В браузере откройте **http://localhost:8000**. Остановка сервера: **Ctrl+C**.

**Вариант вручную** (без скрипта):

```bash
cd audio-mastering-web/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

## Быстрый старт

### 1. Перейти в каталог backend

```bash
cd audio-mastering-web/backend
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# или: venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Запустить сервер

```bash
# из каталога backend
python run.py
```

Или через uvicorn напрямую:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Открыть в браузере

- Интерфейс: **http://localhost:8000**
- API документация: **http://localhost:8000/docs**

## Разделение фронтенда (CSS и JS в отдельные файлы)

Сейчас интерфейс использует **внешний** `app.js` и **встроенные** стили в `index.html`. Чтобы вынести стили в `styles.css`:

1. **Из Ubuntu или WSL** выполните в корне проекта:
   ```bash
   cd frontend && python3 extract_assets.py
   ```
   Будут созданы `styles.css` и обновлён `app.js`.

2. В `index.html` замените блок `<style>...</style>` на:
   ```html
   <link rel="stylesheet" href="styles.css">
   ```

Если не запускать скрипт, страница работает со встроенными стилями и внешним `app.js`.

## Структура проекта

```
audio-mastering-web/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py      # настройки (переменные окружения)
│   │   ├── main.py        # FastAPI: маршруты и раздача фронтенда
│   │   └── pipeline.py    # конвейер мастеринга (DC, EQ, компрессор, лимитер, LUFS)
│   ├── requirements.txt
│   └── run.py             # запуск uvicorn
├── frontend/
│   ├── index.html         # одностраничный интерфейс
│   ├── app.js             # логика (вынесен из index.html)
│   ├── styles.css         # стили (создаётся скриптом, см. ниже)
│   └── extract_assets.py  # скрипт выноса CSS/JS в отдельные файлы
└── README.md
```

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MAGIC_MASTER_MAX_UPLOAD_MB` | Максимальный размер загружаемого файла (МБ) | 100 |
| `MAGIC_MASTER_TEMP_DIR` | Каталог для временных файлов | /tmp/masterflow |
| `MAGIC_MASTER_DEFAULT_TARGET_LUFS` | Целевая громкость по умолчанию (LUFS) | -14.0 |

## API (кратко)

- `GET /api/health` — проверка живости сервиса (для деплоя/мониторинга).
- `GET /api/progress` — содержимое PROGRESS.md (статус выполнения плана разработки).
- `GET /api/presets` — список пресетов LUFS.
- `POST /api/measure` — загрузка файла, ответ: текущая громкость в LUFS.
- `POST /api/master` — загрузка файла + параметры `target_lufs`, `out_format` (и опционально `preset`); ответ — job_id, далее статус и скачивание результата.
- `POST /api/v2/master` — мастеринг по цепочке модулей: форма с `file`, опционально `config` (JSON-цепочка), `target_lufs`, `style`, `out_format`; без `config` используется цепочка по умолчанию. Результат — тот же job_id и те же эндпоинты статуса/результата.
- `GET /api/v2/chain/default` — список модулей цепочки по умолчанию (style, target_lufs); для UI «Модули цепочки» и будущего drag-and-drop.
- `POST /api/v2/analyze` — анализ загруженного файла: LUFS, peak_dbfs, duration_sec, sample_rate, channels; для стерео — stereo_correlation. При `extended=true`: дополнительно `spectrum_bars`, `lufs_timeline`, `timeline_step_sec`; для стерео — `vectorscope_points` (до 1000 точек [l, r] для векторскопа).

Подробнее: **http://localhost:8000/docs**.

## Самодиагностика

Проверка конвейера по стандартам Sony/Warner (LUFS, True Peak, отсутствие NaN):

```bash
cd audio-mastering-web/backend
./venv/bin/python run_self_diagnosis.py
```

С опцией — проверка своего файла и целевого LUFS:

```bash
./venv/bin/python run_self_diagnosis.py /путь/к/файлу.wav -14
```

## Развёртывание (production)

**Пошаговая инструкция по размещению на веб-сервере (по IP или домену):** см. **[DEPLOY.md](DEPLOY.md)**.

Кратко:
1. Собрать архив: `./pack_for_deploy.sh` → скопировать архив на сервер.
2. На сервере: установить FFmpeg и Python 3.10+, распаковать, создать venv в `backend`, установить зависимости.
3. Запуск через systemd (шаблон в `deploy/systemd/`) или вручную: `./venv/bin/python run_production.py`.
4. Опционально: nginx + HTTPS (шаблон в `deploy/nginx/`).

Пример unit-файла systemd (пути заменить на свои):

```ini
[Unit]
Description=Magic Master API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/audio-mastering-web/backend
ExecStart=/path/to/audio-mastering-web/backend/venv/bin/python run_production.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

**Резюме**: веб-версия разворачивается из папки `audio-mastering-web`: установить FFmpeg, Python 3.10+, установить зависимости в `backend`, запустить `python run.py` и открыть http://localhost:8000.
