# История изменений (Magic Master)

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/). Версия приложения задаётся в `backend/app/version.py`.

---

## [0.1.0] — 2026-02-28

### Добавлено

- Веб-интерфейс: загрузка файла (WAV, MP3, FLAC), кнопка «Magic Master», экспорт в WAV/MP3/FLAC.
- Модульный бэкенд v2: цепочка процессоров (DC offset, peak guard, EQ, dynamics, LUFS, exciter, imager, reverb, maximizer).
- Жанровые пресеты: Stream, EDM, Hip-Hop, Classical, Podcast, Lo-fi, House (Ozone 5–style).
- Анализ: LUFS, peak dBFS, спектр (FFT), график LUFS по времени, векторскоп, стерео-корреляция.
- Цепочка модулей в UI: перетаскивание порядка, Amount, phase mode (min/linear), reverb type, M/S, oversampling.
- Экспорт: дизеринг (TPDF, ns_e, ns_itu), обрезка тишины в конце.
- API: `/api`, `/api/health`, `/api/version`, `/api/progress`, `/api/presets`, `/api/styles`, `/api/v2/analyze`, `/api/v2/master`, `/api/v2/chain/default`, статус и результат задачи.
- Страница «Статус плана» (`/progress.html`), документация API (`/docs`).
- Версионность: единый источник в `backend/app/version.py`, отображение в футере и в API.
- Тесты API (pytest), скрипт `backend/run_tests.sh`.

### Документация

- README, DEPLOY, ROADMAP, PROGRESS, CHANGELOG, документ по уровням функций (без регистрации / подписка).

## [0.2.0] — 2026-02-28 (портал)

### Добавлено

- Лендинг-страница `landing.html`: hero с waveform-анимацией и drag-drop зоной «один клик», How it works, Features (Free/Pro), Pricing (Free/Pro/Studio с annual toggle), Testimonials, FAQ, CTA, Footer.
- Маршрутизация: `/` → landing.html, `/app` → mastering app, `/pricing` → pricing.html (FastAPI FileResponse).
- Передача файла с лендинга в приложение через sessionStorage (base64 + redirect).
- Ссылка «← На главную» в шапке приложения.
- Тарифные планы в документе `doc/ФУНКЦИИ_И_УРОВНИ.md` отражены в UI.

---

[0.1.0]: https://github.com/your-repo/audio-mastering-web/releases/tag/v0.1.0
