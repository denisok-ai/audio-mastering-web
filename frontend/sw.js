/**
 * Magic Master — Service Worker (P49)
 * Стратегия: Cache-First для статики, Network-First для API.
 * При офлайн — возвращает кэшированную главную страницу.
 */

const CACHE_VERSION = 'magic-master-v1';
const CACHE_STATIC  = CACHE_VERSION + '-static';
const CACHE_PAGES   = CACHE_VERSION + '-pages';

// Файлы, кэшируемые при первой установке
const PRECACHE_URLS = [
  '/app',
  '/app.js',
  '/manifest.json',
];

// ─── Install ──────────────────────────────────────────────────────────────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_STATIC).then((cache) => {
      return cache.addAll(PRECACHE_URLS).catch(() => {
        // Не фатально — продолжаем без предзагрузки
      });
    }).then(() => self.skipWaiting())
  );
});

// ─── Activate ─────────────────────────────────────────────────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => key.startsWith('magic-master-') && key !== CACHE_STATIC && key !== CACHE_PAGES)
          .map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

// ─── Fetch ────────────────────────────────────────────────────────────────────
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Только GET-запросы
  if (request.method !== 'GET') return;

  // API-запросы: Network-First (без кэша для SSE, медиа и файлов)
  if (url.pathname.startsWith('/api/')) {
    // SSE и бинарные ответы — всегда напрямую
    if (
      url.pathname.startsWith('/api/master/progress/') ||
      url.pathname.startsWith('/api/master/result/') ||
      url.pathname.startsWith('/api/master/preview/')
    ) {
      return; // браузер обрабатывает напрямую
    }
    event.respondWith(
      fetch(request)
        .catch(() => new Response(
          JSON.stringify({ detail: 'Офлайн — данные недоступны' }),
          { status: 503, headers: { 'Content-Type': 'application/json' } }
        ))
    );
    return;
  }

  // Статика (JS, CSS, шрифты) — Cache-First
  if (
    url.pathname.endsWith('.js') ||
    url.pathname.endsWith('.css') ||
    url.pathname.endsWith('.woff2') ||
    url.pathname.endsWith('.woff') ||
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.ico') ||
    url.pathname.endsWith('.svg')
  ) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_STATIC).then((c) => c.put(request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // HTML-страницы — Stale-While-Revalidate
  if (request.headers.get('Accept')?.includes('text/html')) {
    event.respondWith(
      caches.open(CACHE_PAGES).then(async (cache) => {
        const cached = await cache.match(request);
        const fetchPromise = fetch(request).then((response) => {
          if (response.ok) cache.put(request, response.clone());
          return response;
        }).catch(() => null);
        return cached || fetchPromise || new Response(
          '<h1>Офлайн</h1><p>Нет подключения. <a href="/app">Попробуйте снова</a></p>',
          { headers: { 'Content-Type': 'text/html; charset=utf-8' } }
        );
      })
    );
  }
});
