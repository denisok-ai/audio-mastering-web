# План доработок: убрать упоминания сторонних плагинов (iZotope Ozone и др.) в блоке «Цепочка модулей» и связанном UI

**Дата:** 2026-03  
**Цель:** убрать все упоминания известных плагинов (iZotope, Ozone 5 и т.п.) из пользовательского интерфейса  
**Статус:** п. 1–7 выполнены (бэкенд, фронт, лендинг, документация и комментарии). Упоминания iZotope/Ozone убраны из блока «Цепочка модулей», карточек стилей, прогресса мастеринга, ROADMAP, PIPELINE_AUDIO_QUALITY и backend-комментариев.

---

## 1. Блок «Цепочка модулей» — названия модулей (источник: API)

**Файл:** [backend/app/routers/mastering.py](backend/app/routers/mastering.py)

Словарь **CHAIN_MODULE_LABELS** отдаёт названия модулей в UI цепочки. Сейчас в названиях фигурирует «Ozone 5».

| Ключ | Текущий текст | Заменить на |
|------|----------------|------------|
| `target_curve` | Студийный EQ (Ozone 5 Equalizer) | **Студийный EQ** |
| `dynamics` | Многополосная динамика (Ozone 5 Dynamics) | **Многополосная динамика** |
| `exciter` | Гармонический эксайтер (Ozone 5 Exciter) | **Гармонический эксайтер** |
| `imager` | Стерео-расширение (Ozone 5 Imager) | **Стерео-расширение** |

Остальные ключи (`dc_offset`, `peak_guard`, `maximizer`, `normalize_lufs`, `final_spectral_balance`, `style_eq`, `reverb`) уже без упоминаний плагинов — не трогать.

---

## 2. Прогресс мастеринга (шаги пайплайна) — статичная разметка

**Файл:** [frontend/index.html](frontend/index.html)

Элементы с классом `pipe-step` и `pipe-step-label` показывают шаги конвейера. Заменить тексты:

| Текущий текст | Заменить на |
|----------------|------------|
| Студийный EQ (Ozone 5 Equalizer) | **Студийный EQ** |
| Многополосная динамика (Ozone 5 Dynamics) | **Многополосная динамика** |
| Гармонический эксайтер (Ozone 5 Exciter) | **Гармонический эксайтер** |
| Стерео-расширение (Ozone 5 Imager) | **Стерео-расширение** |

Строки ориентировочно: 2461, 2466, 2481, 2486.

---

## 3. Карточка стиля «House» — бейдж «Ozone 5»

**Файл:** [frontend/index.html](frontend/index.html)

- В карточке стиля `data-style="house_basic"` есть блок:  
  `<div class="sc-ozone-badge">Ozone 5</div>` (строка ~2189).
- **Вариант А:** заменить текст на нейтральный, например **«Exciter + Imager»** или **«Расширенная цепочка»**, класс можно переименовать в `sc-pro-badge` или оставить `sc-ozone-badge` (только текст поменять).
- **Вариант Б:** убрать бейдж совсем, если он не нужен.

В том же файле комментарий в CSS (строка ~1009):  
`/* House badge — Ozone 5 Exciter + Imager активны */`  
и класс `.sc-ozone-badge` — при смене текста на «Exciter + Imager» комментарий заменить на:  
`/* House badge — Exciter + Imager активны */`.  
При полном удалении бейджа — удалить и комментарий, и блок с классом.

---

## 4. Функция и вызовы updateOzoneSteps

**Файлы:** [frontend/app.js](frontend/app.js), [frontend/index.html](frontend/index.html) (если там есть дубликат)

- Функция **updateOzoneSteps(style)** только показывает/скрывает шаги «Эксайтер» и «Имаджер» в прогресс-баре в зависимости от стиля (house_basic). Упоминания Ozone в логике нет.
- **Задача:** переименовать в нейтральное имя, например **updatePipelineSteps** или **updateChainSteps**, и обновить все вызовы:
  - в **app.js**: определение функции, вызовы в setFile, resetAll, loadChainModules, при смене стиля, в авто-мастеринге и т.д. (строки по grep: 1043, 1116, 1156, 1184, 1424, 1909, 2112);
  - в **index.html**: если есть копия функции и вызовы (строки ~3294, 3338, 3371, 3396, 3467) — переименовать там же.

Имена **updatePipelineSteps** / **updateChainSteps** не содержат отсылок к плагинам.

---

## 5. Сообщения прогресса на бэкенде (report при мастеринге)

**Файл:** [backend/app/pipeline.py](backend/app/pipeline.py)

В **run_mastering_pipeline** в вызовах `report(...)` используются строки с «Ozone 5»:

| Текущая строка | Заменить на |
|----------------|------------|
| `"Студийный EQ (Ozone 5 Equalizer)"` | **"Студийный EQ"** |
| `"Многополосная динамика и максимайзер (Ozone 5 Dynamics / Maximizer)"` | **"Многополосная динамика и максимайзер"** |
| `f"Гармонический эксайтер (Ozone 5 Exciter) · +{exciter_db:.1f} dB"` | **f"Гармонический эксайтер · +{exciter_db:.1f} dB"** |
| `f"Стерео-расширение (Ozone 5 Imager) · width={imager_width:.2f}"` | **f"Стерео-расширение · width={imager_width:.2f}"** |

Ориентировочно строки: 1697, 1701, 1725, 1729. В docstring функции в начале (строка ~1668) упоминание «iZotope Ozone 5» можно заменить на нейтральное «студийный мастеринг» или оставить только во внутренней документации (см. п. 7).

---

## 6. Лендинг

**Файл:** [frontend/landing.html](frontend/landing.html)

Строка ~777:

- **Было:** «Каждый модуль вдохновлён архитектурой iZotope Ozone 5. Цепочка обработки настраивается вручную.»
- **Стало (вариант):** «Профессиональная цепочка модулей. Порядок и параметры настраиваются вручную.»  
  или: «Цепочка обработки настраивается вручную — порядок модулей и параметры под ваш материал.»

---

## 7. Документация и комментарии в коде (опционально)

По желанию убрать или смягчить упоминания в:

- **[doc/PIPELINE_AUDIO_QUALITY.md](doc/PIPELINE_AUDIO_QUALITY.md)** — строки с «Ozone 5 Equalizer», «Ozone 5 Dynamics/Maximizer» заменить на «Студийный EQ», «Многополосная динамика и максимайзер».
- **[ROADMAP.md](ROADMAP.md)** — таблицы и текст с «iZotope Ozone 5», «Ozone 5-inspired», «Ozone 5 Exciter/Imager» и т.д. Заменить на нейтральные формулировки («студийный мастеринг», «цепочка модулей», «эксайтер/имаджер»).
- **Комментарии в backend:** [backend/app/pipeline.py](backend/app/pipeline.py) (STYLE_CONFIGS, apply_harmonic_exciter, apply_deesser и др.), [backend/app/modules/imaging.py](backend/app/modules/imaging.py), [backend/app/modules/exciter.py](backend/app/modules/exciter.py) — везде, где в комментариях написано «аналог iZotope Ozone 5 …», заменить на «модуль …» без названия плагина или оставить только техническое описание.

Раздел 7 можно выполнять после 1–6, чтобы не смешивать пользовательский интерфейс и внутреннюю/разработческую документацию.

---

## 8. Сводка изменений по файлам

| Файл | Что сделать |
|------|-------------|
| **backend/app/routers/mastering.py** | В CHAIN_MODULE_LABELS убрать «(Ozone 5 …)» у target_curve, dynamics, exciter, imager. |
| **backend/app/pipeline.py** | В report() и при необходимости в docstring убрать «Ozone 5» / «iZotope Ozone 5». |
| **frontend/index.html** | Заменить тексты в pipe-step-label; бейдж «Ozone 5» → нейтральный текст или удалить; комментарий CSS; переименовать updateOzoneSteps → updatePipelineSteps (если дублируется в index.html). |
| **frontend/app.js** | Переименовать updateOzoneSteps → updatePipelineSteps (или updateChainSteps), обновить все вызовы. |
| **frontend/landing.html** | Убрать «iZotope Ozone 5» из описания возможностей. |
| **doc/PIPELINE_AUDIO_QUALITY.md**, **ROADMAP.md**, комментарии в backend | По желанию — нейтральные формулировки (п. 7). |

---

## 9. Порядок выполнения

1. Бэкенд: mastering.py (CHAIN_MODULE_LABELS), pipeline.py (report).
2. Фронт: index.html (pipe-step-label, бейдж House, CSS/комментарий, при наличии — updateOzoneSteps в index.html).
3. Фронт: app.js — переименование updateOzoneSteps и вызовов.
4. Лендинг: landing.html.
5. По желанию: документация и комментарии (п. 7).

После выполнения пользователь не увидит упоминаний iZotope/Ozone в блоке «Цепочка модулей», в шагах прогресса мастеринга, на карточке House и на лендинге.
