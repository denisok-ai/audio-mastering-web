# Эталон для новых публичных SEO-страниц (Magic Master)

Копируйте блоки в новые файлы `frontend/*.html`. Проект — **MPA без сборки**: каждая страница самодостаточна.

## `<head>` — обязательно

1. **Google Analytics** (`G-4TQWW30KHQ`) — как в `landing.html`.
2. **Clarity**: `<script async src="/analytics/clarity.js"></script>`
3. **Meta**: `charset`, `viewport`, уникальные `title`, `description`, `canonical`, `robots`, Open Graph, Twitter Card.
4. **Шрифты**: preconnect + `Space Grotesk` / `JetBrains Mono` по необходимости.
5. **Стили**: inline `<style>` с `:root` токенами как на `landing.html` (`--bg`, `--surface`, `--accent`, …).

## `<body>` — начало

1. Блок **Яндекс.Метрики** `108281088` (как в `landing.html`, сразу после `<body>`).
2. Фон: `.bg-grid` / `.bg-glow` или `.bg` по образцу страницы.
3. **`<nav>`**: логотип на `/`, ссылки на `/app`, `/pricing`, `/blog`, `/tools/lufs-analyzer`, якоря при необходимости, переключатель языка `mm-lang-switch`, CTA «Мастерить».

## Футер

- Колонки ссылок, `id="footVersion"` + `fetch('/api/version')`.
- Юридическая строка как на лендинге.

## Локализация

```html
<script src="/i18n.js"></script>
<script>MagicMasterI18n.init({ pageTitleKey: 'site.seo_page_title' });</script>
```

Ключи — в `frontend/locales/site-ru.json` и `site-en.json`.

## Маршрут FastAPI

В `backend/app/main.py` добавить:

```python
@app.get("/your-path", include_in_schema=False)
async def your_page():
    p = _frontend / "your-page.html"
    return FileResponse(str(p)) if p.is_file() else HTMLResponse("...", 404)
```

## Sitemap

Новый публичный URL — в `frontend/sitemap.xml`.

## Аналитика целей

На CTA: `ym(108281088, 'reachGoal', 'cta_click', { ... });` при наличии `ym`.
