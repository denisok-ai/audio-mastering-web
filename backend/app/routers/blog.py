# @file routers/blog.py
# @description SEO-блог: Markdown + YAML frontmatter из content/blog/, SSR HTML.

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["blog"])

_BLOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "content" / "blog"
_BASE_URL = "https://magicmaster.pro"


def _parse_post(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3 and yaml is not None:
            meta = yaml.safe_load(parts[1]) or {}
            body_md = parts[2].lstrip("\n")
        elif len(parts) >= 3:
            meta, body_md = {}, parts[2].lstrip("\n")
        else:
            meta, body_md = {}, raw
    else:
        meta, body_md = {}, raw

    try:
        from markdown import Markdown

        md = Markdown(extensions=["tables", "fenced_code", "nl2br"])
        body_html = md.convert(body_md)
    except ImportError:
        body_html = "<pre>" + html.escape(body_md) + "</pre>"
    slug = meta.get("slug") or path.stem
    title = meta.get("title") or slug
    desc = meta.get("description") or ""
    date = meta.get("date") or ""
    return {
        "slug": slug,
        "title": title,
        "description": desc,
        "date": date,
        "tags": meta.get("tags") or [],
        "author": meta.get("author") or "Magic Master",
        "image": meta.get("image") or "",
        "schema_type": meta.get("schema_type") or "Article",
        "body_html": body_html,
        "source_path": path,
    }


def _list_posts() -> list[dict[str, Any]]:
    if not _BLOG_DIR.is_dir():
        return []
    posts = []
    for p in sorted(_BLOG_DIR.glob("*.md")):
        try:
            posts.append(_parse_post(p))
        except Exception as e:  # noqa: BLE001
            logger.warning("blog skip %s: %s", p, e)
    posts.sort(key=lambda x: str(x.get("date") or ""), reverse=True)
    return posts


def _shell(
    title: str,
    description: str,
    canonical: str,
    inner: str,
    og_image: str = "",
    extra_head: str = "",
) -> str:
    og = og_image or f"{_BASE_URL}/og-image.png"
    esc_title = html.escape(title)
    esc_desc = html.escape(description)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc_title}</title>
  <meta name="description" content="{esc_desc}">
  <link rel="canonical" href="{html.escape(canonical)}">
  <meta property="og:type" content="website">
  <meta property="og:url" content="{html.escape(canonical)}">
  <meta property="og:title" content="{esc_title}">
  <meta property="og:description" content="{esc_desc}">
  <meta property="og:image" content="{html.escape(og)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{esc_title}">
  <meta name="twitter:description" content="{esc_desc}">
  <meta name="twitter:image" content="{html.escape(og)}">
  <link rel="icon" type="image/png" sizes="192x192" href="/icons/icon-192.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-4TQWW30KHQ"></script>
  <script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-4TQWW30KHQ');</script>
  <script async src="/analytics/clarity.js"></script>
  {extra_head}
  <style>
    :root {{ --bg:#040408; --text:#eeeef8; --soft:#8888aa; --accent:#7c3aed; --border:rgba(255,255,255,.08); --r:12px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:'Space Grotesk',system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.65; }}
    .wrap {{ max-width:720px; margin:0 auto; padding:2rem 1.25rem 4rem; }}
    .nav-top {{ display:flex; gap:1rem; flex-wrap:wrap; align-items:center; margin-bottom:2rem; font-size:.9rem; }}
    .nav-top a {{ color:var(--soft); text-decoration:none; }}
    .nav-top a:hover {{ color:var(--text); }}
    article h1 {{ font-size:clamp(1.6rem,4vw,2.2rem); line-height:1.2; margin:0 0 .75rem; }}
    .meta {{ color:var(--soft); font-size:.88rem; margin-bottom:2rem; }}
    article .body :is(h2,h3) {{ margin-top:1.75rem; }}
    article .body a {{ color:var(--accent); }}
    article .body table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:.9rem; }}
    article .body th, article .body td {{ border:1px solid var(--border); padding:.5rem .65rem; }}
    .cta-box {{ margin-top:3rem; padding:1.5rem; border:1px solid var(--border); border-radius:var(--r); background:rgba(124,58,237,.08); }}
    .cta-box a {{ display:inline-block; margin-top:.75rem; padding:.6rem 1.2rem; background:linear-gradient(135deg,#7c3aed,#a855f7); color:#fff; border-radius:var(--r); text-decoration:none; font-weight:600; }}
    #footVersion {{ margin-top:3rem; font-size:.75rem; color:#555; }}
  </style>
</head>
<body>
<script>(function(m,e,t,r,i,k,a){{m[i]=m[i]||function(){{(m[i].a=m[i].a||[]).push(arguments)}};m[i].l=1*new Date();for(var j=0;j<document.scripts.length;j++){{if(document.scripts[j].src===r){{return;}}}}k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)}})(window,document,'script','https://mc.yandex.ru/metrika/tag.js?id=108281088','ym');ym(108281088,'init',{{ssr:true,webvisor:true,clickmap:true,ecommerce:"dataLayer",referrer:document.referrer,url:location.href,accurateTrackBounce:true,trackLinks:true}});</script>
<noscript><div><img src="https://mc.yandex.ru/watch/108281088" style="position:absolute;left:-9999px;" alt=""/></div></noscript>
<div class="wrap">
  <nav class="nav-top" aria-label="Навигация">
    <a href="/">← Главная</a>
    <a href="/app">Мастеринг</a>
    <a href="/blog">Блог</a>
    <a href="/tools/lufs-analyzer">LUFS-анализатор</a>
  </nav>
  {inner}
  <div id="footVersion"></div>
</div>
<script>
fetch('/api/version').then(function(r){{return r.ok?r.json():null}}).then(function(d){{
  if(!d) return;
  var el=document.getElementById('footVersion');
  if(el) el.textContent = (d.version?'v'+d.version:'') + (d.build_date?' · '+d.build_date:'');
}});
</script>
</body>
</html>"""


@router.get("/blog", response_class=HTMLResponse, include_in_schema=False)
@router.get("/blog/", response_class=HTMLResponse, include_in_schema=False)
def blog_index():
    posts = _list_posts()
    items = []
    for p in posts:
        slug = html.escape(p["slug"])
        title = html.escape(p["title"])
        date = html.escape(str(p.get("date") or ""))
        desc = html.escape((p.get("description") or "")[:180])
        items.append(
            f'<li style="margin-bottom:1.25rem;"><a href="/blog/{slug}" style="color:var(--accent);font-weight:600;">{title}</a>'
            f'<div style="color:var(--soft);font-size:.85rem;">{date}</div>'
            f'<div style="font-size:.9rem;margin-top:.35rem;">{desc}…</div></li>'
        )
    ul = "<ul style='list-style:none;padding:0;margin:0;'>" + "".join(items) + "</ul>"
    inner = (
        "<article><h1>Блог Magic Master</h1>"
        "<p class='meta'>Гайды по мастерингу, LUFS и AI-музыке.</p>"
        + ul
        + "</article>"
    )
    html_page = _shell(
        "Блог — Magic Master",
        "Статьи о мастеринге, LUFS, Suno, подкастах и стриминге.",
        f"{_BASE_URL}/blog",
        inner,
    )
    return HTMLResponse(content=html_page)


@router.get("/blog/{slug}", response_class=HTMLResponse, include_in_schema=False)
def blog_post(slug: str):
    path = _BLOG_DIR / f"{slug}.md"
    if not path.is_file():
        raise HTTPException(404, "Статья не найдена")
    post = _parse_post(path)
    canonical = f"{_BASE_URL}/blog/{slug}"
    article_schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": post["title"],
        "description": post["description"],
        "datePublished": str(post.get("date") or ""),
        "author": {"@type": "Organization", "name": post.get("author") or "Magic Master"},
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
    }
    extra = f'<script type="application/ld+json">{json.dumps(article_schema, ensure_ascii=False)}</script>'
    inner = f"""<article>
  <h1>{html.escape(post['title'])}</h1>
  <div class="meta">{html.escape(str(post.get('date') or ''))} · {html.escape(post.get('author') or '')}</div>
  <div class="body">{post['body_html']}</div>
  <div class="cta-box">
    <strong>Попробуйте Magic Master</strong>
    <p style="margin:.5rem 0 0;color:var(--soft);font-size:.9rem;">Загрузите трек — готовый мастер за секунды.</p>
    <a href="/app">Открыть мастеринг →</a>
    <a href="/tools/lufs-analyzer" style="margin-left:.5rem;background:transparent;border:1px solid var(--border);color:var(--text);">LUFS-анализатор</a>
  </div>
</article>"""
    return HTMLResponse(
        content=_shell(
            f"{post['title']} | Magic Master",
            post["description"] or post["title"],
            canonical,
            inner,
            og_image=post.get("image") or "",
            extra_head=extra,
        )
    )
