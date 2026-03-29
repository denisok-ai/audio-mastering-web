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
<html lang="ru" id="htmlRoot">
<head>
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-4TQWW30KHQ"></script>
  <script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-4TQWW30KHQ');</script>
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
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script async src="/analytics/clarity.js"></script>
  {extra_head}
  <style>
    :root {{
      --bg:#040408; --surface:rgba(10,10,18,0.96); --border:rgba(255,255,255,0.06);
      --border2:rgba(255,255,255,0.12); --text:#eeeef8; --soft:#8888aa; --dim:#44445e;
      --accent:#7c3aed; --acc2:#a855f7; --acc-glow:rgba(124,58,237,0.4);
      --cyan:#06b6d4; --green:#10b981; --r:12px; --r-lg:20px; --r-xl:30px;
    }}
    *,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
    html {{ scroll-behavior:smooth; }}
    body {{ font-family:'Space Grotesk',system-ui,sans-serif; background:var(--bg); color:var(--text); overflow-x:hidden; line-height:1.65; }}
    .bg-grid {{
      position:fixed; inset:0; pointer-events:none; z-index:0;
      background-image:linear-gradient(rgba(108,75,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(108,75,255,0.03) 1px,transparent 1px);
      background-size:48px 48px;
    }}
    .bg-glow {{ position:fixed; pointer-events:none; z-index:0; border-radius:50%; filter:blur(120px); opacity:0.18; }}
    .bg-glow-1 {{ width:700px; height:500px; background:var(--accent); top:-100px; left:-100px; }}
    .bg-glow-2 {{ width:500px; height:500px; background:var(--acc2); top:30%; right:-100px; }}
    nav {{
      position:sticky; top:0; z-index:100;
      display:flex; align-items:center; justify-content:space-between;
      padding:0 2rem; height:64px;
      background:rgba(6,6,16,0.88);
      backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
      border-bottom:1px solid var(--border);
    }}
    .nav-logo {{ display:flex; align-items:center; gap:.55rem; text-decoration:none; color:var(--text); }}
    .nav-logo-icon {{
      width:32px; height:32px; border-radius:8px;
      background:linear-gradient(135deg,var(--accent),var(--acc2));
      display:flex; align-items:center; justify-content:center;
      box-shadow:0 0 16px var(--acc-glow);
    }}
    .nav-logo-icon svg {{ width:16px; height:16px; color:#fff; }}
    .nav-logo-name {{ font-size:1.05rem; font-weight:700; letter-spacing:-.01em; }}
    .nav-logo-badge {{ font-size:.6rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; padding:2px 6px; border-radius:4px; background:linear-gradient(135deg,var(--accent),var(--acc2)); color:#fff; margin-left:2px; }}
    .nav-links {{ display:flex; align-items:center; gap:.25rem; }}
    .nav-link {{ padding:.45rem .9rem; border-radius:var(--r); color:var(--soft); font-size:.9rem; text-decoration:none; transition:color .15s, background .15s; }}
    .nav-link:hover {{ color:var(--text); background:rgba(255,255,255,0.06); }}
    .nav-link.active {{ color:var(--text); background:rgba(108,75,255,0.12); }}
    .nav-cta {{ padding:.5rem 1.2rem; border-radius:var(--r); background:linear-gradient(135deg,var(--accent),var(--acc2)); color:#fff; font-size:.9rem; font-weight:600; text-decoration:none; transition:opacity .15s, transform .15s; box-shadow:0 0 20px var(--acc-glow); }}
    .nav-cta:hover {{ opacity:.9; transform:translateY(-1px); }}
    .mm-lang-switch {{ display:inline-flex; align-items:center; gap:.35rem; margin-right:.5rem; font-size:.85rem; color:var(--soft); }}
    .mm-lang-switch a {{ color:var(--soft); text-decoration:none; font-weight:600; padding:.2rem .35rem; border-radius:6px; cursor:pointer; }}
    .mm-lang-switch a:hover {{ color:var(--text); }}
    .mm-lang-switch a.mm-lang-active {{ color:var(--text); text-decoration:underline; }}
    @media (max-width:600px) {{ .nav-link {{ display:none; }} nav {{ padding:0 1rem; }} }}
    .page-wrap {{ position:relative; z-index:1; max-width:780px; margin:0 auto; padding:2.5rem 1.5rem 3rem; }}
    article h1 {{ font-size:clamp(1.6rem,4vw,2.2rem); font-weight:800; line-height:1.2; margin:0 0 .75rem; letter-spacing:-.02em; }}
    .meta {{ color:var(--soft); font-size:.88rem; margin-bottom:2rem; }}
    article .body :is(h2,h3) {{ margin-top:1.75rem; font-weight:700; }}
    article .body a {{ color:var(--acc2); }}
    article .body table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:.9rem; }}
    article .body th, article .body td {{ border:1px solid var(--border); padding:.5rem .65rem; }}
    article .body code {{ background:rgba(124,58,237,.12); padding:.15rem .35rem; border-radius:4px; font-size:.88em; font-family:'JetBrains Mono',monospace; }}
    article .body pre {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--r); padding:1rem; overflow-x:auto; margin:1rem 0; }}
    article .body pre code {{ background:none; padding:0; }}
    article .body img {{ max-width:100%; height:auto; border-radius:var(--r); margin:1rem 0; }}
    article .body blockquote {{ border-left:3px solid var(--accent); padding:.5rem 1rem; margin:1rem 0; color:var(--soft); }}
    .cta-box {{
      margin-top:3rem; padding:2rem; border-radius:var(--r-xl);
      background:linear-gradient(135deg,rgba(108,75,255,0.18),rgba(168,85,247,0.1),rgba(34,211,238,0.08));
      border:1px solid rgba(108,75,255,0.3); text-align:center;
      box-shadow:0 0 60px rgba(108,75,255,0.15);
    }}
    .cta-box strong {{ font-size:1.1rem; }}
    .cta-box a {{
      display:inline-block; margin-top:.75rem; padding:.7rem 1.5rem;
      background:linear-gradient(135deg,var(--accent),var(--acc2)); color:#fff;
      border-radius:var(--r); text-decoration:none; font-weight:700;
      box-shadow:0 0 24px var(--acc-glow); transition:opacity .15s, transform .15s;
    }}
    .cta-box a:hover {{ opacity:.9; transform:translateY(-1px); }}
    .blog-list {{ list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:1rem; }}
    .blog-list li {{
      padding:1.25rem; border-radius:var(--r-lg);
      background:var(--surface); border:1px solid var(--border);
      transition:transform .2s, box-shadow .2s;
    }}
    .blog-list li:hover {{ transform:translateY(-2px); box-shadow:0 8px 32px rgba(108,75,255,0.1); }}
    .blog-list a {{ color:var(--acc2); font-weight:700; text-decoration:none; font-size:1.05rem; }}
    .blog-list a:hover {{ text-decoration:underline; }}
    footer {{
      position:relative; z-index:1; border-top:1px solid var(--border);
      padding:3rem 2rem 2rem; display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr;
      gap:2rem; max-width:980px; margin:0 auto;
    }}
    .foot-brand {{ display:flex; flex-direction:column; gap:.75rem; }}
    .foot-logo {{ display:flex; align-items:center; gap:.5rem; text-decoration:none; color:var(--text); }}
    .foot-logo-icon {{ width:28px; height:28px; border-radius:7px; background:linear-gradient(135deg,var(--accent),var(--acc2)); display:flex; align-items:center; justify-content:center; }}
    .foot-logo-icon svg {{ width:13px; height:13px; color:#fff; }}
    .foot-logo-name {{ font-weight:700; font-size:.95rem; }}
    .foot-desc {{ font-size:.82rem; color:var(--dim); line-height:1.6; max-width:220px; }}
    .foot-version {{ font-size:.72rem; color:var(--dim); font-family:'JetBrains Mono',monospace; }}
    .foot-col-title {{ font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); margin-bottom:.85rem; }}
    .foot-col a {{ display:block; font-size:.85rem; color:var(--soft); text-decoration:none; margin-bottom:.55rem; transition:color .15s; }}
    .foot-col a:hover {{ color:var(--text); }}
    .foot-bottom {{
      position:relative; z-index:1; max-width:980px; margin:0 auto;
      padding:1.25rem 2rem; border-top:1px solid var(--border);
      display:flex; justify-content:space-between; align-items:center;
      font-size:.78rem; color:var(--dim); flex-wrap:wrap; gap:.5rem;
    }}
    @media (max-width:700px) {{
      footer {{ grid-template-columns:1fr 1fr; }}
      .foot-brand {{ grid-column:1/-1; }}
      .foot-bottom {{ flex-direction:column; text-align:center; }}
    }}
    @media (max-width:480px) {{ footer {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<script>(function(m,e,t,r,i,k,a){{m[i]=m[i]||function(){{(m[i].a=m[i].a||[]).push(arguments)}};m[i].l=1*new Date();for(var j=0;j<document.scripts.length;j++){{if(document.scripts[j].src===r){{return;}}}}k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)}})(window,document,'script','https://mc.yandex.ru/metrika/tag.js?id=108281088','ym');ym(108281088,'init',{{ssr:true,webvisor:true,clickmap:true,ecommerce:"dataLayer",referrer:document.referrer,url:location.href,accurateTrackBounce:true,trackLinks:true}});</script>
<noscript><div><img src="https://mc.yandex.ru/watch/108281088" style="position:absolute;left:-9999px;" alt=""/></div></noscript>
<div class="bg-grid"></div>
<div class="bg-glow bg-glow-1"></div>
<div class="bg-glow bg-glow-2"></div>

<nav>
  <a href="/" class="nav-logo">
    <div class="nav-logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
    </div>
    <span class="nav-logo-name">Magic Master</span>
    <span class="nav-logo-badge">Pro</span>
  </a>
  <div class="nav-links">
    <a href="/app" class="nav-link" data-i18n="landing.foot_mastering">Мастеринг</a>
    <a href="/blog" class="nav-link active" data-i18n="landing.nav_blog">Блог</a>
    <a href="/tools/lufs-analyzer" class="nav-link" data-i18n="landing.nav_lufs">LUFS</a>
    <a href="/suno-mastering" class="nav-link" data-i18n="landing.nav_suno">Suno</a>
    <a href="/pricing" class="nav-link" data-i18n="landing.nav_pricing">Тарифы</a>
    <span class="mm-lang-switch" aria-label="Language">
      <a href="#" data-mm-lang="ru" data-i18n="common.lang_ru">RU</a>
      <span>|</span>
      <a href="#" data-mm-lang="en" data-i18n="common.lang_en">EN</a>
    </span>
    <a href="/app" class="nav-cta" data-i18n="landing.nav_cta">Мастерить бесплатно →</a>
  </div>
</nav>

<div class="page-wrap">
  {inner}
</div>

<footer>
  <div class="foot-brand">
    <a href="/" class="foot-logo">
      <div class="foot-logo-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
      </div>
      <span class="foot-logo-name">Magic Master</span>
    </a>
    <div class="foot-desc" data-i18n="landing.foot_desc">Профессиональный мастеринг аудио прямо в браузере. Индустриальные стандарты без подписки на DAW.</div>
    <div class="foot-version" id="footVersion"></div>
  </div>
  <div class="foot-col">
    <div class="foot-col-title" data-i18n="landing.foot_col_product">Продукт</div>
    <a href="/app" data-i18n="landing.foot_mastering">Мастеринг</a>
    <a href="/pricing" data-i18n="landing.nav_pricing">Тарифы</a>
    <a href="/blog" data-i18n="landing.foot_blog">Блог</a>
    <a href="/tools/lufs-analyzer" data-i18n="landing.foot_lufs">LUFS-анализатор</a>
    <a href="/suno-mastering" data-i18n="landing.foot_suno">Мастеринг для Suno</a>
    <a href="/telegram-bot" data-i18n="landing.foot_telegram">Telegram-бот</a>
    <a href="/referral" data-i18n="landing.foot_referral">Реферальная программа</a>
  </div>
  <div class="foot-col">
    <div class="foot-col-title" data-i18n="landing.foot_col_dev">Разработка</div>
    <a href="/docs" data-i18n="landing.foot_api_docs">API Документация</a>
    <a href="/progress.html" data-i18n="landing.foot_plan_status">Статус плана</a>
    <a href="/api/health" data-i18n="landing.foot_svc_status">Статус сервиса</a>
    <a href="/api/version" data-i18n="landing.foot_api_ver">Версия API</a>
  </div>
  <div class="foot-col">
    <div class="foot-col-title" data-i18n="landing.foot_col_support">Поддержка</div>
    <a href="mailto:support@magicmaster.app" data-i18n="landing.foot_write_us">Написать нам</a>
    <a href="/pricing" data-i18n="landing.foot_pricing_pay">Тарифы и оплата</a>
  </div>
</footer>
<div class="foot-bottom">
  <span data-i18n="landing.foot_bottom1">© 2026 Magic Master. Профессиональный мастеринг аудио.</span>
  <span data-i18n="landing.foot_bottom2">Сделано с ♥ для музыкантов и продюсеров</span>
</div>
<div style="max-width:980px;margin:0 auto;padding:0.5rem 2rem 1rem;font-size:.68rem;color:var(--dim);text-align:center;position:relative;z-index:1;" data-i18n="landing.foot_legal">
  Все упомянутые торговые марки и названия продуктов принадлежат их правообладателям. Magic Master не аффилирован с указанными компаниями.
</div>

<script src="/i18n.js"></script>
<script>if(typeof MagicMasterI18n!=='undefined')MagicMasterI18n.init({{}});</script>
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
            f'<li><a href="/blog/{slug}">{title}</a>'
            f'<div style="color:var(--soft);font-size:.85rem;margin-top:.25rem;">{date}</div>'
            f'<div style="font-size:.9rem;color:var(--soft);margin-top:.35rem;">{desc}…</div></li>'
        )
    ul = '<ul class="blog-list">' + "".join(items) + "</ul>"
    inner = (
        "<article><h1>Блог Magic Master</h1>"
        '<p class="meta">Гайды по мастерингу, LUFS и AI-музыке.</p>'
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
