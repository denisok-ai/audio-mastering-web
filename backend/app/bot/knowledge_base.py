"""Bilingual product knowledge base for AI consultant (RU + EN).

Single source of truth for LLM context in Telegram bot and web chat.
"""

PRODUCT_KNOWLEDGE_RU = """\
## О продукте

Magic Master (magicmaster.pro) — онлайн-сервис профессионального мастеринга аудио.
Загрузите трек → выберите стиль → получите готовый мастер за секунды.
Поддержка форматов: WAV, MP3, FLAC. Результат: 24-bit WAV.
Идеально подходит для AI-сгенерированной музыки (Suno, Udio и др.).

## Стили мастеринга

| Стиль | LUFS | Описание |
|-------|------|----------|
| standard | −14 | Стриминг (Spotify, Apple Music, YouTube Music) |
| edm | −9 | Электронная музыка, клубный саунд |
| hiphop | −13 | Хип-хоп, трэп, R&B |
| classical | −18 | Классика, камерная музыка, оркестр |
| podcast | −16 | Подкасты, голосовой контент |
| lofi | −18 | Lo-fi, винтажный, тёплый звук |
| house_basic | −10 | Хаус, тек-хаус, клубный буст |
| dry_vocal | −14 | Сухой вокал, ровная АЧХ |

## Платформенные пресеты

- Spotify: −14 LUFS, True Peak ≤ −1 dBTP
- Apple Music: −16 LUFS
- YouTube: −14 LUFS
- Club: −9 LUFS (максимальная громкость)
- Broadcast: −24 LUFS (ТВ/радио стандарт EBU R128)

## PRO-модули (уникальные функции)

1. **Spectral Denoiser** — удаление шума методом Wiener-фильтрации. Параметр: denoise_strength (0–1).
2. **De-esser** — подавление сибилянтов (s/sh/ц). Параметр: deesser_threshold (−30…−10 dB).
3. **Transient Designer** — контроль атаки и сустейна ударных/перкуссии. Параметры: transient_attack, transient_sustain.
4. **Parallel Compression** — нью-йоркская техника: сильно сжатый сигнал подмешивается к оригиналу. Параметр: parallel_mix (0–1).
5. **Dynamic EQ** — 8-полосный EQ, реагирующий на уровень сигнала. Включение: dynamic_eq_enabled.
6. **Reference Match** — подгонка спектра под загруженный эталонный трек.

## Цепочка обработки (pipeline)

DC offset → Peak Guard → Style EQ (5-band) → Multiband Dynamics (4-band) →
Exciter → Stereo Imager → [PRO: Denoiser → De-esser → Transients → Parallel Comp → Dyn EQ] →
Maximizer → LUFS Normalization → Dithering (16/24-bit)

## Тарифы

### Free ($0)
- До 3 мастерингов в день
- Все стили доступны
- Без регистрации
- Формат: WAV до 100 МБ
- 5 AI-запросов в день

### Pro ($9/мес, $7/мес при оплате за год)
- Безлимитные мастеринги
- Все PRO-модули
- WAV/MP3/FLAC до 300 МБ
- 50 AI-запросов в день
- Приоритетная очередь

### Studio ($29/мес, $24/мес при оплате за год)
- Всё из Pro
- Batch-обработка до 50 файлов
- Файлы до 800 МБ
- Безлимитный AI
- API-доступ
- Приоритетная обработка

Все PRO-модули доступны бесплатно в период бета-тестирования.

## FAQ

**Какие форматы поддерживаются?**
WAV, MP3, FLAC. Результат всегда 24-bit WAV.

**Безопасно ли загружать файлы?**
Да. Файлы обрабатываются на сервере и автоматически удаляются через 1 час.

**Что такое LUFS?**
LUFS (Loudness Units Full Scale) — международный стандарт измерения воспринимаемой громкости. −14 LUFS — стандарт стриминговых платформ.

**Как подготовить трек после AI-генерации (Suno, Udio)?**
1. Экспортируйте в WAV (не MP3) для максимального качества.
2. Оставьте headroom: пиковый уровень не выше −3 dBFS.
3. Не применяйте лимитер/нормализацию перед загрузкой.
4. Выберите стиль по жанру (edm, hiphop, standard и т.д.).
5. При необходимости используйте PRO-модули (деноайзер для шума, де-эссер для свистящих).

**Чем Magic Master лучше других сервисов мастеринга?**
- 6 уникальных PRO-модулей (нет у конкурентов)
- Reference Match — подгонка спектра под эталон
- Бесплатное использование без регистрации
- Оптимизация для AI-сгенерированной музыки
- Telegram-бот для мастеринга на ходу

**Можно ли использовать API?**
Да, REST API доступен в Studio тарифе.

**Как работает мастеринг в Telegram?**
Отправьте аудиофайл боту → выберите стиль → получите мастер-версию обратно.

**Что такое True Peak?**
True Peak (dBTP) — межвыборочный пик. Стандарт: ≤ −1 dBTP, чтобы избежать клиппинга при конвертации.

**Как привязать Telegram к аккаунту сайта?**
Команда /link email@example.com → введите код из письма: /code 123456

## Команды Telegram-бота

- /start — начало работы
- /master — отправьте файл для мастеринга
- /analyze — анализ громкости и спектра
- /presets — список стилей
- /ask вопрос — AI-консультант
- /balance — токены и тариф
- /history — последние мастеринги
- /pricing — тарифы
- /status — статус сервиса
- /link email — привязка к аккаунту
- /lang — смена языка (RU/EN)
- /ref — пригласительная ссылка

## Мягкие конверсионные фразы

- «Попробуйте мастеринг прямо сейчас — это бесплатно! Отправьте /master»
- «На Pro тарифе доступны все PRO-модули без ограничений. Подробнее: /pricing»
- «Загрузите свой трек на magicmaster.pro — результат за секунды!»
"""

PRODUCT_KNOWLEDGE_EN = """\
## About the Product

Magic Master (magicmaster.pro) — online professional audio mastering service.
Upload a track → choose a style → get a finished master in seconds.
Supported formats: WAV, MP3, FLAC. Output: 24-bit WAV.
Perfect for AI-generated music (Suno, Udio, etc.).

## Mastering Styles

| Style | LUFS | Description |
|-------|------|-------------|
| standard | −14 | Streaming (Spotify, Apple Music, YouTube Music) |
| edm | −9 | Electronic music, club sound |
| hiphop | −13 | Hip-hop, trap, R&B |
| classical | −18 | Classical, chamber music, orchestra |
| podcast | −16 | Podcasts, voice content |
| lofi | −18 | Lo-fi, vintage, warm sound |
| house_basic | −10 | House, tech-house, club boost |
| dry_vocal | −14 | Dry vocal, flat frequency response |

## Platform Presets

- Spotify: −14 LUFS, True Peak ≤ −1 dBTP
- Apple Music: −16 LUFS
- YouTube: −14 LUFS
- Club: −9 LUFS (maximum loudness)
- Broadcast: −24 LUFS (TV/radio standard EBU R128)

## PRO Modules (Unique Features)

1. **Spectral Denoiser** — noise removal via Wiener filtering. Parameter: denoise_strength (0–1).
2. **De-esser** — sibilance suppression (s/sh sounds). Parameter: deesser_threshold (−30…−10 dB).
3. **Transient Designer** — attack and sustain control for drums/percussion. Parameters: transient_attack, transient_sustain.
4. **Parallel Compression** — New York technique: heavily compressed signal blended with the original. Parameter: parallel_mix (0–1).
5. **Dynamic EQ** — 8-band EQ that reacts to signal level. Toggle: dynamic_eq_enabled.
6. **Reference Match** — spectrum matching to a loaded reference track.

## Processing Chain (Pipeline)

DC offset → Peak Guard → Style EQ (5-band) → Multiband Dynamics (4-band) →
Exciter → Stereo Imager → [PRO: Denoiser → De-esser → Transients → Parallel Comp → Dyn EQ] →
Maximizer → LUFS Normalization → Dithering (16/24-bit)

## Pricing

### Free ($0)
- Up to 3 masterings per day
- All styles available
- No registration required
- Format: WAV up to 100 MB
- 5 AI queries per day

### Pro ($9/mo, $7/mo billed annually)
- Unlimited masterings
- All PRO modules
- WAV/MP3/FLAC up to 300 MB
- 50 AI queries per day
- Priority queue

### Studio ($29/mo, $24/mo billed annually)
- Everything in Pro
- Batch processing up to 50 files
- Files up to 800 MB
- Unlimited AI
- API access
- Priority processing

All PRO modules are free during the beta period.

## FAQ

**What formats are supported?**
WAV, MP3, FLAC. Output is always 24-bit WAV.

**Is it safe to upload files?**
Yes. Files are processed on the server and automatically deleted after 1 hour.

**What is LUFS?**
LUFS (Loudness Units Full Scale) — the international standard for measuring perceived loudness. −14 LUFS is the streaming platform standard.

**How to prepare a track after AI generation (Suno, Udio)?**
1. Export as WAV (not MP3) for maximum quality.
2. Leave headroom: peak level no higher than −3 dBFS.
3. Do not apply a limiter/normalization before uploading.
4. Choose a style by genre (edm, hiphop, standard, etc.).
5. Use PRO modules if needed (denoiser for noise, de-esser for sibilance).

**Why is Magic Master better than other mastering services?**
- 6 unique PRO modules (not available elsewhere)
- Reference Match — spectrum matching to a reference
- Free to use without registration
- Optimized for AI-generated music
- Telegram bot for mastering on the go

**Can I use the API?**
Yes, REST API is available in the Studio tier.

**How does Telegram mastering work?**
Send an audio file to the bot → choose a style → receive the mastered version back.

**What is True Peak?**
True Peak (dBTP) — inter-sample peak. Standard: ≤ −1 dBTP to avoid clipping during conversion.

**How to link Telegram to a website account?**
Command /link email@example.com → enter the code from the email: /code 123456

## Telegram Bot Commands

- /start — get started
- /master — send a file for mastering
- /analyze — loudness & spectrum analysis
- /presets — list of styles
- /ask question — AI consultant
- /balance — tokens & tier
- /history — recent masterings
- /pricing — plans
- /status — service status
- /link email — link to account
- /lang — change language (RU/EN)
- /ref — invite link

## Soft Conversion Phrases

- "Try mastering right now — it's free! Send /master"
- "With the Pro plan you get all PRO modules with no limits. Details: /pricing"
- "Upload your track at magicmaster.pro — results in seconds!"
"""


def get_product_knowledge(lang: str = "ru") -> str:
    """Return product knowledge base for the given language."""
    if (lang or "ru").lower().startswith("en"):
        return PRODUCT_KNOWLEDGE_EN
    return PRODUCT_KNOWLEDGE_RU
