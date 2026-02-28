# @file llm_guard.py
# @description Защита от LLM-инъекций: валидация и санитизация пользовательского ввода перед отправкой в LLM.
# Настройки через system_settings (админка): llm_guard_enabled, llm_guard_max_length_*, llm_guard_forbidden_* и др.

import re
from typing import Optional

# Контексты проверки: разный лимит длины и строгость для чата и NL→config
CONTEXT_CHAT = "chat"
CONTEXT_NL_CONFIG = "nl_config"

# Фразы по умолчанию, часто используемые в prompt injection (англ. и рус.)
_DEFAULT_FORBIDDEN_SUBSTRINGS = [
    "ignore previous",
    "ignore all",
    "ignore above",
    "disregard",
    "forget your",
    "forget the",
    "you are now",
    "new instructions",
    "system prompt",
    "reveal your",
    "output your",
    "переведи на английский и выполни",
    "игнорируй предыдущие",
    "забудь инструкции",
    "ты теперь",
    "выведи системный промпт",
    "покажи инструкции",
]


def _get_guard_setting(key: str, default=None):
    """Читает настройку защиты из settings_store."""
    try:
        from . import settings_store
        return settings_store.get_setting(key)
    except Exception:  # noqa: BLE001
        return default


def _get_forbidden_list() -> list[str]:
    """Список запрещённых подстрок (при совпадении ввод блокируется)."""
    raw = _get_guard_setting("llm_guard_forbidden_substrings")
    if raw is None:
        return _DEFAULT_FORBIDDEN_SUBSTRINGS
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if x]
    if isinstance(raw, str) and raw.strip():
        try:
            import json
            lst = json.loads(raw)
            return [str(x).strip().lower() for x in lst if x] if isinstance(lst, list) else _DEFAULT_FORBIDDEN_SUBSTRINGS
        except Exception:  # noqa: BLE001
            pass
    return _DEFAULT_FORBIDDEN_SUBSTRINGS


def _get_forbidden_regex() -> Optional[re.Pattern]:
    """Скомпилированный regex для блокировки (если задан в настройках)."""
    raw = _get_guard_setting("llm_guard_forbidden_regex")
    if not raw or not str(raw).strip():
        return None
    try:
        return re.compile(str(raw).strip(), re.IGNORECASE | re.DOTALL)
    except re.error:
        return None


def _get_max_length(context: str) -> int:
    """Максимальная длина пользовательского ввода в символах. 0 = без ограничения."""
    if context == CONTEXT_CHAT:
        v = _get_guard_setting("llm_guard_max_length_chat")
    else:
        v = _get_guard_setting("llm_guard_max_length_nl")
    if v is None:
        return 2000 if context == CONTEXT_CHAT else 500
    try:
        n = int(v)
        return max(0, n)
    except (TypeError, ValueError):
        return 2000 if context == CONTEXT_CHAT else 500


def _sanitize_text(text: str) -> str:
    """Удаление управляющих символов и лишних переводов строк."""
    if not text:
        return ""
    # Удаляем управляющие символы (кроме \n, \r, \t)
    cleaned = "".join(c for c in text if c in "\n\r\t" or (ord(c) >= 32 and ord(c) != 127))
    # Нормализуем переводы строк
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
    return cleaned.strip()


def validate_llm_input(
    text: str,
    context: str = CONTEXT_CHAT,
) -> tuple[bool, Optional[str], str]:
    """
    Проверка пользовательского ввода перед отправкой в LLM.

    Returns:
        (allowed, sanitized_text, error_message)
        - allowed=False: ввод заблокирован (инъекция или нарушение правил), sanitized_text=None, error_message — причина.
        - allowed=True: sanitized_text — очищенная/обрезанная строка, error_message пустой.
    """
    try:
        from . import settings_store
        if not settings_store.get_setting_bool("llm_guard_enabled", True):
            sanitized = _sanitize_text(text or "")
            return True, sanitized, ""
    except Exception:  # noqa: BLE001
        pass

    if not text or not str(text).strip():
        return True, "", ""

    raw = str(text).strip()
    sanitized = _sanitize_text(raw)
    lower = sanitized.lower()

    # 1. Запрещённые подстроки
    for phrase in _get_forbidden_list():
        if phrase and phrase in lower:
            return False, None, "Запрос отклонён: обнаружено недопустимое содержимое."

    # 2. Запрещённый regex
    pattern = _get_forbidden_regex()
    if pattern and pattern.search(sanitized):
        return False, None, "Запрос отклонён: содержимое не прошло проверку."

    # 3. Ограничение длины (обрезаем или отклоняем в зависимости от настройки)
    max_len = _get_max_length(context)
    if max_len > 0 and len(sanitized) > max_len:
        try:
            from . import settings_store
            truncate = settings_store.get_setting_bool("llm_guard_truncate_on_long", True)
        except Exception:  # noqa: BLE001
            truncate = True
        if truncate:
            sanitized = sanitized[:max_len].rstrip()
        else:
            return False, None, f"Запрос слишком длинный. Максимум {max_len} символов."

    return True, sanitized, ""


def validate_chat_message(content: str, role: str) -> tuple[bool, Optional[str], str]:
    """
    Проверка одного сообщения в чате.
    role: если пользователь передал "system" — принудительно считаем "user" (защита от подмены роли).
    """
    try:
        from . import settings_store
        if not settings_store.get_setting_bool("llm_guard_block_role_system", True):
            pass
        else:
            if (role or "").strip().lower() == "system":
                return False, None, "Недопустимая роль сообщения."
    except Exception:  # noqa: BLE001
        if (role or "").strip().lower() == "system":
            return False, None, "Недопустимая роль сообщения."

    return validate_llm_input(content or "", CONTEXT_CHAT)
