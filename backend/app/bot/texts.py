"""Тексты RU/EN для бота."""

T = {
    "ru": {
        "start_welcome": (
            "🎛 <b>Magic Master</b> — автоматический мастеринг.\n\n"
            "Привяжите аккаунт сайта: <code>/link ваш@email.ru</code>\n"
            "Затем отправьте аудио или нажмите кнопки ниже.\n"
            "Канал: @PROmagicmaster"
        ),
        "help": (
            "<b>Команды</b>\n"
            "/master — мастеринг (пришлите файл после команды)\n"
            "/analyze — анализ громкости и спектра\n"
            "/presets — стили мастеринга\n"
            "/ask &lt;вопрос&gt; — AI про мастеринг\n"
            "/balance — токены и тариф\n"
            "/history — последние мастеринги\n"
            "/pricing — тарифы\n"
            "/status — статус сервиса\n"
            "/link email — привязка к аккаунту\n"
            "/lang — язык\n"
            "/ref — пригласить друга"
        ),
        "not_linked": "Сначала привяжите аккаунт: <code>/link email@example.com</code> — код придёт на почту.",
        "blocked": "Аккаунт заблокирован. Обратитесь в поддержку.",
        "free_limit": "Лимит Free: 1 мастеринг в неделю. Тарифы: /pricing",
        "no_tokens": "Недостаточно токенов. /pricing",
        "daily_cap": "Дневной лимит тарифа исчерпан. Попробуйте завтра.",
        "file_too_large": "Файл больше {mb} МБ — загрузите через сайт: {url}/app",
        "bad_format": "Нужен WAV, MP3, FLAC или голосовое сообщение.",
        "send_audio": "Пришлите аудиофайл (до 20 МБ) или голосовое.",
        "choose_preset": "Выберите стиль мастеринга:",
        "processing": "⏳ Мастеринг… подождите.",
        "done": "✅ Готово!\nДо: {before:.1f} LUFS → После: {after:.1f} LUFS",
        "error": "Ошибка обработки. Попробуйте другой файл или сайт.",
        "analyze_result": (
            "📊 <b>Анализ</b>\n"
            "LUFS: {lufs}\nPeak: {peak} dBFS\nДлительность: {dur} с\nКаналы: {ch}\nSR: {sr} Hz"
        ),
        "balance": "Тариф: <b>{tier}</b>\nТокены: <b>{tokens}</b>",
        "history_empty": "История пуста.",
        "history_line": "• {name} — {style}, {lufs} LUFS",
        "pricing": "Тарифы и оплата: {url}/pricing",
        "status_ok": "Статус: <b>{st}</b>\nЗадач в работе: {jobs}",
        "link_sent": "Код отправлен на {email}. Введите: <code>/code 123456</code>",
        "link_no_user": "Пользователь с таким email не найден. Зарегистрируйтесь на сайте.",
        "link_bad_email": "Укажите email: <code>/link you@mail.ru</code>",
        "code_ok": "Аккаунт привязан: {email}",
        "code_bad": "Неверный или просроченный код.",
        "unlink_ok": "Telegram отвязан от аккаунта.",
        "unlink_none": "Аккаунт не был привязан.",
        "lang_ok": "Язык: {lang}",
        "ref": "Пригласить друга: {url}/register?ref=tg_{uid}",
        "admin_only": "Команда только для администратора.",
        "broadcast_usage": "Использование: ответьте на это сообщение текстом или: /broadcast текст",
        "broadcast_done": "Отправлено: {n} пользователям.",
        "ai_disabled": "AI временно недоступен.",
        "ai_limit": "Лимит AI на сегодня исчерпан.",
        "welcome_d1": "👋 Напоминание: попробуйте мастеринг — отправьте WAV/MP3 в чат или /master",
        "welcome_d3": "💡 Совет: после генерации музыки LLM мастеринг выравнивает громкость и баланс.",
        "welcome_d7": "🎵 Ваш трек ждёт обработки — /master",
        "weekly_dm": "📈 За неделю мастерингов: {n}. Токены: {tokens}. Сайт: {url}/app",
    },
    "en": {
        "start_welcome": (
            "🎛 <b>Magic Master</b> — automatic mastering.\n\n"
            "Link your account: <code>/link you@email.com</code>\n"
            "Then send audio or use buttons below.\n"
            "Channel: @PROmagicmaster"
        ),
        "help": (
            "<b>Commands</b>\n"
            "/master — mastering (send a file)\n"
            "/analyze — loudness & spectrum\n"
            "/presets — mastering styles\n"
            "/ask &lt;question&gt; — AI assistant\n"
            "/balance — tokens & tier\n"
            "/history — recent jobs\n"
            "/pricing — plans\n"
            "/status — service status\n"
            "/link email — link account\n"
            "/lang — language\n"
            "/ref — invite link"
        ),
        "not_linked": "Link your account first: <code>/link email@example.com</code> — code by email.",
        "blocked": "Account blocked.",
        "free_limit": "Free tier: 1 mastering per week. Plans: /pricing",
        "no_tokens": "Not enough tokens. /pricing",
        "daily_cap": "Daily tier limit reached. Try tomorrow.",
        "file_too_large": "File over {mb} MB — use the website: {url}/app",
        "bad_format": "Send WAV, MP3, FLAC or a voice message.",
        "send_audio": "Send an audio file (up to 20 MB) or voice.",
        "choose_preset": "Choose mastering style:",
        "processing": "⏳ Mastering… please wait.",
        "done": "✅ Done!\nBefore: {before:.1f} LUFS → After: {after:.1f} LUFS",
        "error": "Processing error. Try another file or the website.",
        "analyze_result": (
            "📊 <b>Analysis</b>\n"
            "LUFS: {lufs}\nPeak: {peak} dBFS\nDuration: {dur} s\nChannels: {ch}\nSR: {sr} Hz"
        ),
        "balance": "Tier: <b>{tier}</b>\nTokens: <b>{tokens}</b>",
        "history_empty": "No history yet.",
        "history_line": "• {name} — {style}, {lufs} LUFS",
        "pricing": "Plans: {url}/pricing",
        "status_ok": "Status: <b>{st}</b>\nJobs running: {jobs}",
        "link_sent": "Code sent to {email}. Enter: <code>/code 123456</code>",
        "link_no_user": "No user with this email. Register on the site.",
        "link_bad_email": "Usage: <code>/link you@mail.com</code>",
        "code_ok": "Linked: {email}",
        "code_bad": "Invalid or expired code.",
        "unlink_ok": "Telegram unlinked.",
        "unlink_none": "Account was not linked.",
        "lang_ok": "Language: {lang}",
        "ref": "Invite: {url}/register?ref=tg_{uid}",
        "admin_only": "Admin only.",
        "broadcast_usage": "Reply with text or: /broadcast your message",
        "broadcast_done": "Sent to {n} users.",
        "ai_disabled": "AI is temporarily disabled.",
        "ai_limit": "Daily AI limit reached.",
        "welcome_d1": "👋 Try mastering — send WAV/MP3 or /master",
        "welcome_d3": "💡 Tip: after LLM music generation, mastering balances loudness.",
        "welcome_d7": "🎵 Your track is waiting — /master",
        "weekly_dm": "📈 Masterings this week: {n}. Tokens: {tokens}. {url}/app",
    },
}


def lang_for_user(user_row) -> str:
    if user_row and getattr(user_row, "telegram_lang", None):
        l = (user_row.telegram_lang or "ru").lower()[:2]
        return l if l in T else "ru"
    return "ru"


def txt(lang: str, key: str, **kwargs) -> str:
    l = lang if lang in T else "ru"
    s = T[l].get(key) or T["ru"].get(key, key)
    return s.format(**kwargs) if kwargs else s
