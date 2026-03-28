/**
 * Magic Master — единая локализация (RU/EN).
 * Ключ localStorage: magic_lang (миграция с magic_master_lang, magic_locale).
 * Приоритет: ?lang= > magic_lang > CIS по navigator.language > иначе en.
 */
(function (global) {
  var STORAGE_KEY = 'magic_lang';
  var LEGACY_KEYS = ['magic_master_lang', 'magic_locale'];
  /** ISO 639-1: языки СНГ и соседних рынков → интерфейс по умолчанию RU */
  var CIS_LANGS = {
    ru: 1, uk: 1, be: 1, kk: 1, uz: 1, ky: 1, tg: 1, az: 1, hy: 1, ka: 1,
    ro: 1 /* MD часто ro */,
  };

  function migrateLegacy() {
    try {
      if (localStorage.getItem(STORAGE_KEY)) return;
      for (var i = 0; i < LEGACY_KEYS.length; i++) {
        var v = localStorage.getItem(LEGACY_KEYS[i]);
        if (v) {
          localStorage.setItem(STORAGE_KEY, String(v).toLowerCase() === 'en' ? 'en' : 'ru');
          break;
        }
      }
    } catch (e) { /* ignore */ }
  }

  function browserPrimaryLang() {
    var nav = global.navigator;
    var raw = (nav.language || (nav.languages && nav.languages[0]) || 'en');
    return String(raw).split('-')[0].toLowerCase();
  }

  function resolveLangFromUrl() {
    var m = /[?&]lang=(\w+)/i.exec(global.location.search);
    if (!m) return null;
    return m[1].toLowerCase() === 'en' ? 'en' : 'ru';
  }

  function resolveLang() {
    var fromUrl = resolveLangFromUrl();
    if (fromUrl) return fromUrl;
    migrateLegacy();
    try {
      var s = localStorage.getItem(STORAGE_KEY);
      if (s === 'en' || s === 'ru') return s;
    } catch (e2) { /* ignore */ }
    return CIS_LANGS[browserPrimaryLang()] ? 'ru' : 'en';
  }

  function persistLang(lang) {
    lang = lang === 'en' ? 'en' : 'ru';
    try {
      localStorage.setItem(STORAGE_KEY, lang);
      for (var i = 0; i < LEGACY_KEYS.length; i++) {
        try { localStorage.removeItem(LEGACY_KEYS[i]); } catch (e) { /* ignore */ }
      }
    } catch (e) { /* ignore */ }
    return lang;
  }

  var currentLang = 'ru';

  function loadLocale(lang) {
    lang = lang === 'en' ? 'en' : 'ru';
    var base = '/locales/' + lang + '.json';
    var site = '/locales/site-' + lang + '.json';
    return Promise.all([
      fetch(base).then(function (r) { return r.ok ? r.json() : {}; }),
      fetch(site).then(function (r) { return r.ok ? r.json() : {}; }).catch(function () { return {}; }),
    ]).then(function (parts) {
      global.__locale = Object.assign({}, parts[0], parts[1]);
      return global.__locale;
    }).catch(function () {
      global.__locale = {};
      return global.__locale;
    });
  }

  function __t(key) {
    return (global.__locale && global.__locale[key]) || key;
  }

  function applyI18n(root) {
    root = root || document;
    root.querySelectorAll('[data-i18n]').forEach(function (el) {
      if (el.id === 'refFileName' && global.__refFile) return;
      var key = el.getAttribute('data-i18n');
      if (key && __t(key) !== key) el.textContent = __t(key);
    });
    root.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      var key = el.getAttribute('data-i18n-html');
      if (key && __t(key) !== key) el.innerHTML = __t(key);
    });
    root.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      var key = el.getAttribute('data-i18n-placeholder');
      if (key && __t(key) !== key) el.setAttribute('placeholder', __t(key));
    });
    root.querySelectorAll('[data-i18n-aria-label]').forEach(function (el) {
      var key = el.getAttribute('data-i18n-aria-label');
      if (key && __t(key) !== key) el.setAttribute('aria-label', __t(key));
    });
    var htmlEl = document.documentElement;
    if (htmlEl) htmlEl.setAttribute('lang', currentLang === 'en' ? 'en' : 'ru');
    var metaDesc = document.querySelector('meta[name="description"][data-i18n-content]');
    if (metaDesc) {
      var dk = metaDesc.getAttribute('data-i18n-content');
      if (dk && __t(dk) !== dk) metaDesc.setAttribute('content', __t(dk));
    }
  }

  function setDocumentTitleFromKey(titleKey) {
    if (titleKey && __t(titleKey) !== titleKey) document.title = __t(titleKey);
  }

  function wireLanguageSwitcher(root) {
    root = root || document;
    root.querySelectorAll('[data-mm-lang]').forEach(function (el) {
      el.addEventListener('click', function (e) {
        e.preventDefault();
        var l = el.getAttribute('data-mm-lang');
        if (l !== 'en' && l !== 'ru') return;
        persistLang(l);
        global.location.reload();
      });
    });
  }

  function highlightActiveLang() {
    document.querySelectorAll('[data-mm-lang]').forEach(function (el) {
      var l = el.getAttribute('data-mm-lang');
      el.classList.toggle('mm-lang-active', l === currentLang);
      if (el.tagName === 'A') el.setAttribute('aria-current', l === currentLang ? 'true' : 'false');
    });
  }

  /**
   * @param {object} opts
   * @param {string} [opts.pageTitleKey] — ключ для document.title
   * @param {boolean} [opts.wireAppLocale=false] — #localeWrap .locale-btn без reload (только app)
   */
  function init(opts) {
    opts = opts || {};
    currentLang = resolveLang();
    persistLang(currentLang);
    global.__t = __t;
    return loadLocale(currentLang).then(function () {
      global.__t = __t;
      setDocumentTitleFromKey(opts.pageTitleKey);
      applyI18n();
      wireLanguageSwitcher();
      highlightActiveLang();
      if (opts.wireAppLocale) {
        var wrap = document.getElementById('localeWrap');
        if (wrap) {
          wrap.querySelectorAll('.locale-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
              var l = btn.getAttribute('data-lang');
              if (l === 'en' || l === 'ru') {
                persistLang(l);
                loadLocale(l).then(function () {
                  applyI18n();
                  wrap.querySelectorAll('.locale-btn').forEach(function (b) {
                    b.classList.toggle('active', b.getAttribute('data-lang') === l);
                  });
                });
              }
            });
          });
          wrap.querySelectorAll('.locale-btn').forEach(function (b) {
            b.classList.toggle('active', b.getAttribute('data-lang') === currentLang);
          });
        }
      }
      try {
        global.dispatchEvent(new CustomEvent('magicmaster:locale', { detail: { lang: currentLang } }));
      } catch (e) { /* ignore */ }
      return currentLang;
    });
  }

  global.MagicMasterI18n = {
    init: init,
    getLang: function () { return currentLang; },
    resolveLang: resolveLang,
    persistLang: persistLang,
    loadLocale: loadLocale,
    applyI18n: applyI18n,
    __t: __t,
    browserPrimaryLang: browserPrimaryLang,
  };
})(typeof window !== 'undefined' ? window : this);
