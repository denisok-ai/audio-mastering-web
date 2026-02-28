# @file settings_store.py
# @description Слой настроек: чтение из БД (переопределения) с fallback на config/.env
# @dependencies database, config

import json
import time
from typing import Any, Optional

from .config import settings as _config

# Ключи, которые можно переопределять через админку (и типы для десериализации)
_SETTING_TYPES: dict[str, str] = {
    "max_upload_mb": "int",
    "allowed_extensions": "json_list",  # список строк
    "temp_dir": "str",
    "default_target_lufs": "float",
    "jobs_done_ttl_seconds": "int",
    "debug_mode": "bool",
    "require_email_verify": "bool",
    "global_rate_limit": "int",
    "cors_origins": "str",
    "smtp_host": "str",
    "smtp_port": "int",
    "smtp_user": "str",
    "smtp_password": "str",
    "smtp_from": "str",
    "smtp_use_tls": "bool",
    "yookassa_shop_id": "str",
    "yookassa_secret_key": "str",
    "yookassa_return_url": "str",
    "yookassa_webhook_ip_whitelist": "str",
    "telegram_bot_token": "str",
    "telegram_admin_chat_id": "str",
    "ai_backend": "str",
    "openai_api_key": "str",
    "anthropic_api_key": "str",
    "deepseek_api_key": "str",
    "deepseek_base_url": "str",
    "deepseek_model": "str",
    "ai_limit_free": "int",
    "ai_limit_pro": "int",
    "ai_limit_studio": "int",
    "feature_ai_enabled": "bool",
    "feature_batch_enabled": "bool",
    "feature_registration_enabled": "bool",
    "notify_email_on_register": "bool",
    "notify_telegram_on_payment": "bool",
    "maintenance_mode": "bool",
    "default_locale": "str",
    # Защита от LLM-инъекций (llm_guard)
    "llm_guard_enabled": "bool",
    "llm_guard_max_length_chat": "int",
    "llm_guard_max_length_nl": "int",
    "llm_guard_forbidden_substrings": "json_list",
    "llm_guard_forbidden_regex": "str",
    "llm_guard_truncate_on_long": "bool",
    "llm_guard_block_role_system": "bool",
}


def _parse_value(raw: str, typ: str) -> Any:
    if typ == "str":
        return raw
    if typ == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0
    if typ == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0
    if typ == "bool":
        return (raw or "").strip().lower() in ("1", "true", "yes", "on")
    if typ == "json_list":
        try:
            lst = json.loads(raw)
            return list(lst) if isinstance(lst, list) else [str(lst)]
        except (json.JSONDecodeError, TypeError):
            return []
    return raw


def _serialize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_setting(key: str) -> Any:
    """Читает настройку: сначала из БД (system_settings), при отсутствии — из config."""
    try:
        from .database import DB_AVAILABLE, SessionLocal, SystemSetting
        if DB_AVAILABLE and SessionLocal is not None and SystemSetting is not None:
            db = SessionLocal()
            try:
                row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
                if row and row.value is not None:
                    typ = _SETTING_TYPES.get(key, "str")
                    return _parse_value(row.value, typ)
            finally:
                db.close()
    except Exception:  # noqa: BLE001
        pass
    # Fallback to config
    if hasattr(_config, key):
        return getattr(_config, key)
    return None


def get_setting_str(key: str) -> str:
    v = get_setting(key)
    if v is None:
        return ""
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def get_setting_int(key: str, default: int = 0) -> int:
    v = get_setting(key)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_setting_float(key: str, default: float = 0.0) -> float:
    v = get_setting(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def get_setting_bool(key: str, default: bool = False) -> bool:
    v = get_setting(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return (str(v).strip().lower() in ("1", "true", "yes", "on"))


def set_setting(key: str, value: Any, admin_id: Optional[int] = None) -> None:
    """Записывает переопределение настройки в БД."""
    try:
        from .database import DB_AVAILABLE, SessionLocal, SystemSetting
        if not DB_AVAILABLE or SessionLocal is None or SystemSetting is None:
            return
        db = SessionLocal()
        try:
            row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
            str_val = _serialize_value(value)
            now = time.time()
            if row:
                row.value = str_val
                row.updated_at = now
                row.updated_by = admin_id
            else:
                db.add(SystemSetting(key=key, value=str_val, updated_at=now, updated_by=admin_id))
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        raise


def get_all_overrides() -> dict[str, str]:
    """Возвращает все переопределения из БД (key -> value string) для админки."""
    out: dict[str, str] = {}
    try:
        from .database import DB_AVAILABLE, SessionLocal, SystemSetting
        if not DB_AVAILABLE or SessionLocal is None or SystemSetting is None:
            return out
        db = SessionLocal()
        try:
            for row in db.query(SystemSetting).all():
                out[row.key] = row.value or ""
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass
    return out


def set_settings_batch(updates: dict[str, Any], admin_id: Optional[int] = None) -> None:
    """Массовое обновление настроек. Ключи должны быть в _SETTING_TYPES."""
    for key, value in updates.items():
        if key in _SETTING_TYPES:
            set_setting(key, value, admin_id)
