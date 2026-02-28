# @file ai.py
# @description AI-агенты: рекомендатор пресетов, отчёт, NL→config, чат. Лимиты по тарифам.
# @dependencies app.config
# @created 2026-02-28

import datetime
import json
import os
from typing import Any, Optional, Tuple

from .config import settings

# Счётчики AI-запросов: ключ (user_id или ip) -> { "count": int, "day": str }
_ai_rate_limits: dict[str, dict] = {}

# Стили, доступные в STYLE_CONFIGS (pipeline)
VALID_STYLES = {"standard", "edm", "hiphop", "classical", "podcast", "lofi", "house_basic"}

# Встроенные промпты (используются, если в config не заданы ai_prompt_*)
DEFAULT_PROMPT_RECOMMEND = """Ты — звукоинженер. По JSON с анализом трека (LUFS, peak_dbfs, duration_sec, channels, spectrum_*) предложи пресет мастеринга.
Доступные стили: standard, edm, hiphop, classical, podcast, lofi, house_basic.
Ответь строго в формате JSON с полями: "style" (один из стилей), "target_lufs" (число, например -14), "reason" (короткая фраза на русском).
Правила: если LUFS сильно ниже -18 и много низких частот — edm или house_basic с target_lufs -9..-10; подкасты/длинный моно — podcast -16; классика — classical -18; по умолчанию standard -14."""

DEFAULT_PROMPT_REPORT = "Ты — звукоинженер. По данным анализа трека напиши краткое описание (2–3 предложения) и 2–3 рекомендации по мастерингу на русском. Ответь строго JSON: {\"summary\": \"...\", \"recommendations\": [\"...\", \"...\"]}."

DEFAULT_PROMPT_NL_CONFIG = """Ты — помощник по настройке мастеринга. Пользователь описывает желаемые изменения (громкость, воздух, сибилянты, басы и т.д.).
Ответь строго JSON с полями: "target_lufs" (число или null, если не меняем), "chain_config" (объект с ключами, например denoise_strength 0-1, deesser_threshold в dB, transient_attack/sustain, parallel_mix, dynamic_eq_enabled).
Только допустимые ключи: target_lufs, denoise_strength, deesser_threshold, deesser_enabled, transient_attack, transient_sustain, transient_enabled, parallel_mix, dynamic_eq_enabled.
Если не можешь извлечь параметры — верни пустой chain_config и target_lufs null."""

DEFAULT_PROMPT_CHAT = "Ты — звукоинженер-консультант по мастерингу. Отвечай кратко на русском. Стили: standard (-14 LUFS), edm (-9), hiphop (-13), classical (-18), podcast (-16), lofi (-18), house_basic (-10). Можешь объяснять параметры цепочки: деноайзер, де-эссер, транзиенты, параллельная компрессия, динамический EQ."

_DEFAULT_BY_SLUG = {
    "recommend": DEFAULT_PROMPT_RECOMMEND,
    "report": DEFAULT_PROMPT_REPORT,
    "nl_config": DEFAULT_PROMPT_NL_CONFIG,
    "chat": DEFAULT_PROMPT_CHAT,
}

_SETTINGS_KEY_BY_SLUG = {
    "recommend": "ai_prompt_recommend",
    "report": "ai_prompt_report",
    "nl_config": "ai_prompt_nl_config",
    "chat": "ai_prompt_chat",
}


def _get_prompt_from_source(slug: str) -> str:
    """Читает промпт: сначала БД (активная версия), затем .env, затем встроенный."""
    try:
        from .database import SessionLocal, get_active_prompt_body
        if SessionLocal:
            db = SessionLocal()
            try:
                body = get_active_prompt_body(db, slug)
                if body:
                    return body
            finally:
                db.close()
    except Exception:  # noqa: BLE001
        pass
    key = _SETTINGS_KEY_BY_SLUG.get(slug)
    if key:
        try:
            from . import settings_store
            s = settings_store.get_setting(key)
            if s and str(s).strip():
                return str(s).strip()
        except Exception:  # noqa: BLE001
            pass
        s = (getattr(settings, key, "") or "").strip()
        if s:
            return s
    return _DEFAULT_BY_SLUG.get(slug, "")


def _get_prompt_recommend() -> str:
    return _get_prompt_from_source("recommend")


def _get_prompt_report() -> str:
    return _get_prompt_from_source("report")


def _get_prompt_nl_config() -> str:
    return _get_prompt_from_source("nl_config")


def _get_prompt_chat() -> str:
    return _get_prompt_from_source("chat")


def get_effective_prompts() -> dict:
    """Текущие промпты (из .env или встроенные). Для отображения в админке."""
    return {
        "recommend": _get_prompt_recommend(),
        "report": _get_prompt_report(),
        "nl_config": _get_prompt_nl_config(),
        "chat": _get_prompt_chat(),
    }


def _get_llm_setting(key: str, default: str = ""):
    """Читает настройку LLM: сначала из админки (БД), затем из config/env."""
    try:
        from . import settings_store
        v = settings_store.get_setting(key)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    except Exception:  # noqa: BLE001
        pass
    return (getattr(settings, key, None) or os.environ.get(key.upper().replace(".", "_"), "") or default).strip()


def _get_llm_client() -> Tuple[Any, str]:
    """
    Возвращает (client, model) для настроенного бэкенда (OpenAI или DeepSeek).
    Клиент совместим с OpenAI API (chat.completions.create). Если ключа нет — (None, "").
    """
    try:
        from openai import OpenAI
    except ImportError:
        return (None, "")

    backend = (_get_llm_setting("ai_backend") or getattr(settings, "ai_backend", "openai")).lower()
    if backend == "deepseek":
        key = _get_llm_setting("deepseek_api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            return (None, "")
        base_url = _get_llm_setting("deepseek_base_url") or "https://api.deepseek.com"
        model = _get_llm_setting("deepseek_model") or "deepseek-chat"
        client = OpenAI(api_key=key, base_url=base_url)
        return (client, model)
    # openai
    key = _get_llm_setting("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return (None, "")
    client = OpenAI(api_key=key)
    return (client, "gpt-4o-mini")


def get_ai_backend_name() -> str:
    """Имя настроенного бэкенда для отображения (openai, deepseek или пусто)."""
    client, model = _get_llm_client()
    if not client or not model:
        return ""
    backend = (_get_llm_setting("ai_backend") or "openai").lower()
    return "deepseek" if backend == "deepseek" else "openai"


def get_ai_limit_for_tier(tier: str) -> int:
    """Дневной лимит AI-запросов для тарифа. -1 = без лимита."""
    try:
        from . import settings_store
        t = (tier or "free").lower()
        if t == "pro":
            return settings_store.get_setting_int("ai_limit_pro", 50)
        if t == "studio":
            return settings_store.get_setting_int("ai_limit_studio", -1)
        return settings_store.get_setting_int("ai_limit_free", 5)
    except Exception:  # noqa: BLE001
        pass
    t = (tier or "free").lower()
    if t in ("pro",):
        return getattr(settings, "ai_limit_pro", 50)
    if t in ("studio",):
        return getattr(settings, "ai_limit_studio", -1)
    return getattr(settings, "ai_limit_free", 5)


def check_ai_rate_limit(identifier: str, tier: str) -> dict:
    """
    Проверка лимита AI для identifier (user_id или ip).
    Возвращает {"ok": bool, "used": int, "limit": int, "remaining": int, "reset_at": str}.
    """
    limit = get_ai_limit_for_tier(tier)
    today = datetime.date.today().isoformat()
    entry = _ai_rate_limits.get(identifier)
    used = entry["count"] if (entry and entry.get("day") == today) else 0
    if limit < 0:
        return {"ok": True, "used": used, "limit": -1, "remaining": -1, "reset_at": today}
    remaining = max(0, limit - used)
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    return {
        "ok": used < limit,
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "reset_at": tomorrow,
    }


def record_ai_usage(identifier: str) -> None:
    """Увеличить счётчик AI-запросов для identifier."""
    today = datetime.date.today().isoformat()
    entry = _ai_rate_limits.get(identifier)
    if not entry or entry.get("day") != today:
        _ai_rate_limits[identifier] = {"count": 1, "day": today}
    else:
        _ai_rate_limits[identifier]["count"] = entry["count"] + 1


def _rule_based_recommend(analysis: dict) -> dict:
    """
    Правило-движок: по LUFS, длительности и спектру предлагает style и target_lufs.
    Без вызова LLM. Используется при отсутствии API ключа или как fallback.
    """
    lufs = analysis.get("lufs")
    duration_sec = analysis.get("duration_sec") or 0
    channels = analysis.get("channels", 1)
    spectrum = analysis.get("spectrum_bars")
    # Упрощённая оценка «много низких»: сумма первых 1/4 баров спектра
    low_freq_energy = 0.0
    if isinstance(spectrum, (list, tuple)) and len(spectrum) >= 4:
        low_freq_energy = sum(spectrum[: max(1, len(spectrum) // 4)]) / max(1, len(spectrum) // 4)

    style = "standard"
    target_lufs = -14.0
    reason = "Стандартный пресет для стриминга (−14 LUFS)."

    if lufs is not None:
        if lufs < -20.0 and low_freq_energy > 0.3:
            style = "edm"
            target_lufs = -9.0
            reason = "Трек тихий и с выраженными низами — предложен EDM (−9 LUFS)."
        elif lufs < -20.0:
            style = "standard"
            target_lufs = -14.0
            reason = "Трек тихий — целевая громкость −14 LUFS (стриминг)."
        elif duration_sec > 600 and channels == 1:
            style = "podcast"
            target_lufs = -16.0
            reason = "Длинный моно-трек — пресет подкаста (−16 LUFS)."
        elif duration_sec < 120:
            style = "standard"
            target_lufs = -14.0
            reason = "Короткий трек — стриминг −14 LUFS."

    return {
        "style": style,
        "target_lufs": target_lufs,
        "chain_config": None,
        "reason": reason,
    }


def recommend_preset(analysis: dict) -> dict:
    """
    По данным анализа возвращает рекомендацию: style, target_lufs, опционально chain_config и reason.
    Использует OpenAI с structured output при наличии ключа, иначе правило-движок.
    """
    # Сжать анализ для промпта: только ключевые поля
    payload = {
        "lufs": analysis.get("lufs"),
        "peak_dbfs": analysis.get("peak_dbfs"),
        "duration_sec": analysis.get("duration_sec"),
        "sample_rate": analysis.get("sample_rate"),
        "channels": analysis.get("channels"),
        "stereo_correlation": analysis.get("stereo_correlation"),
    }
    if "spectrum_bars" in analysis and isinstance(analysis["spectrum_bars"], (list, tuple)):
        bars = analysis["spectrum_bars"]
        if len(bars) >= 8:
            payload["spectrum_low"] = sum(bars[: len(bars) // 4]) / (len(bars) // 4)
            payload["spectrum_mid"] = sum(bars[len(bars) // 4 : 3 * len(bars) // 4]) / max(1, len(bars) // 2)
            payload["spectrum_high"] = sum(bars[3 * len(bars) // 4 :]) / max(1, len(bars) // 4)
    payload_str = json.dumps(payload, ensure_ascii=False)

    client, model = _get_llm_client()
    if client and model:
        try:
            return _recommend_via_llm(client, model, payload_str, analysis)
        except Exception:
            pass
    return _rule_based_recommend(analysis)


def _recommend_via_llm(client: Any, model: str, payload_str: str, analysis: dict) -> dict:
    """Вызов LLM (OpenAI/DeepSeek) с structured output (JSON)."""
    system = _get_prompt_recommend()
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Анализ трека:\n{payload_str}"},
        ],
        "max_tokens": 300,
    }
    # OpenAI и DeepSeek поддерживают response_format для JSON
    try:
        kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        pass
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return _rule_based_recommend(analysis)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if "{" in text and "}" in text:
            i, j = text.find("{"), text.rfind("}") + 1
            if j > i:
                try:
                    data = json.loads(text[i:j])
                except json.JSONDecodeError:
                    pass
    if not data:
        return _rule_based_recommend(analysis)
    style = (data.get("style") or "standard").lower()
    if style not in VALID_STYLES:
        style = "standard"
    target_lufs = float(data.get("target_lufs", -14))
    target_lufs = max(-24, min(-6, target_lufs))
    return {
        "style": style,
        "target_lufs": target_lufs,
        "chain_config": data.get("chain_config"),
        "reason": data.get("reason") or "Рекомендация по анализу трека.",
    }


def report_with_recommendations(analysis: dict) -> dict:
    """
    Текстовый отчёт по анализу + 2–3 рекомендации на русском.
    При наличии OpenAI или DeepSeek — генерируем через LLM; иначе — шаблон по числам.
    """
    client, model = _get_llm_client()
    if client and model:
        try:
            return _report_via_llm(client, model, analysis)
        except Exception:
            pass
    return _report_rule_based(analysis)


def _report_rule_based(analysis: dict) -> dict:
    """Краткий отчёт без LLM."""
    lufs = analysis.get("lufs")
    peak = analysis.get("peak_dbfs")
    dur = analysis.get("duration_sec")
    ch = analysis.get("channels", 1)
    lines = []
    if lufs is not None:
        lines.append(f"Интегрированная громкость: {lufs:.1f} LUFS.")
    if peak is not None:
        lines.append(f"Пик: {peak:.1f} dBFS.")
    if dur is not None:
        lines.append(f"Длительность: {dur:.1f} с. Каналов: {ch}.")
    summary = " ".join(lines) if lines else "Данные анализа отсутствуют."
    recs = []
    if lufs is not None:
        if lufs < -18:
            recs.append("Трек тихий — увеличьте целевую громкость (например −14 LUFS для стриминга).")
        elif lufs > -10:
            recs.append("Трек громкий — для стриминга лучше снизить до −14 LUFS.")
    if not recs:
        recs.append("Используйте пресет по жанру и при необходимости подстройте цепочку модулей.")
    return {"summary": summary, "recommendations": recs}


def _report_via_llm(client: Any, model: str, analysis: dict) -> dict:
    """Генерация отчёта через LLM (OpenAI/DeepSeek)."""
    payload = {
        "lufs": analysis.get("lufs"),
        "peak_dbfs": analysis.get("peak_dbfs"),
        "duration_sec": analysis.get("duration_sec"),
        "channels": analysis.get("channels"),
        "stereo_correlation": analysis.get("stereo_correlation"),
    }
    system = _get_prompt_report()
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "max_tokens": 400,
    }
    try:
        kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        pass
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return _report_rule_based(analysis)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if "{" in text and "}" in text:
            i, j = text.find("{"), text.rfind("}") + 1
            if j > i:
                try:
                    data = json.loads(text[i:j])
                except json.JSONDecodeError:
                    pass
    if not data:
        return _report_rule_based(analysis)
    return {
        "summary": data.get("summary", ""),
        "recommendations": data.get("recommendations") or [],
    }


def nl_to_config(natural_language: str, current_config: Optional[dict] = None) -> dict:
    """
    Преобразование запроса на естественном языке в chain_config и/или target_lufs.
    При отсутствии API (OpenAI/DeepSeek) — возвращаем текущий config без изменений.
    Перед вызовом LLM ввод проверяется через llm_guard (защита от инъекций).
    """
    natural_language = (natural_language or "").strip()
    if not natural_language:
        return {"chain_config": current_config or {}, "target_lufs": None}

    try:
        from .llm_guard import validate_llm_input, CONTEXT_NL_CONFIG
        allowed, sanitized, err = validate_llm_input(natural_language, CONTEXT_NL_CONFIG)
        if not allowed:
            return {"chain_config": current_config or {}, "target_lufs": None, "error": err or "Запрос отклонён."}
        natural_language = sanitized or natural_language
    except Exception:  # noqa: BLE001
        pass

    client, model = _get_llm_client()
    if not client or not model:
        return {"chain_config": current_config or {}, "target_lufs": None}

    system = _get_prompt_nl_config()
    user_msg = f"Текущие настройки: {json.dumps(current_config or {}, ensure_ascii=False)}\nЗапрос пользователя: {natural_language}"
    kwargs = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}], "max_tokens": 350}
    try:
        kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        pass
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return {"chain_config": current_config or {}, "target_lufs": None}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"chain_config": current_config or {}, "target_lufs": None}
    return {
        "chain_config": data.get("chain_config") or current_config or {},
        "target_lufs": data.get("target_lufs"),
    }


def chat_assistant(messages: list[dict], context: Optional[dict] = None) -> str:
    """
    Чат-помощник: история сообщений + контекст (анализ/настройки). Возвращает ответ текстом.
    Поддерживаются бэкенды OpenAI и DeepSeek. Ввод проверяется через llm_guard.
    """
    client, model = _get_llm_client()
    if not client or not model:
        return "Чат-помощник доступен при настройке AI (OpenAI или DeepSeek) в Pro/Studio."

    system = _get_prompt_chat()
    if context:
        system += f"\nКонтекст текущего трека/настроек: {json.dumps(context, ensure_ascii=False)[:800]}"

    try:
        from .llm_guard import validate_chat_message
        use_guard = True
    except Exception:  # noqa: BLE001
        use_guard = False

    openai_messages = [{"role": "system", "content": system}]
    for m in messages[-10:]:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = str(m.get("content", ""))
        if use_guard and role == "user":
            allowed, sanitized, err = validate_chat_message(content, role)
            if not allowed:
                return err or "Запрос отклонён: не прошёл проверку безопасности."
            content = sanitized or content
        elif use_guard and role == "system":
            allowed, _, err = validate_chat_message("", role)
            if not allowed:
                return err or "Недопустимое сообщение."
        openai_messages.append({"role": role, "content": content[:2000]})

    resp = client.chat.completions.create(model=model, messages=openai_messages, max_tokens=500)
    return (resp.choices[0].message.content or "").strip()
