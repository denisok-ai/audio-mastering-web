# @file routers/auth.py
# @description Auth-эндпоинты: регистрация, логин, профиль, история, API-ключи, пресеты.
# @created 2026-03-01

import csv
import datetime
import io
import json
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from ..auth import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from ..database import (
    DB_AVAILABLE,
    check_and_expire_subscription,
    create_api_key,
    create_mastering_record,
    create_saved_preset,
    create_user,
    delete_mastering_record,
    delete_saved_preset,
    get_api_keys_for_user,
    get_db,
    get_saved_preset_by_id,
    get_user_by_email,
    get_user_history,
    get_user_presets,
    get_user_stats,
    revoke_api_key,
)
from ..deps import (
    check_auth_rate_limit,
    get_current_user_optional,
    require_auth_available,
)
from ..helpers import get_client_ip
from ..config import settings

router = APIRouter()

# ─── In-memory токены (верификация + сброс пароля) ───────────────────────────
_verify_tokens: dict[str, dict] = {}
_VERIFY_TOKEN_TTL = 86400  # 24 часа

_reset_tokens: dict[str, dict] = {}
_RESET_TOKEN_TTL = 3600  # 1 час


def _cleanup_verify_tokens() -> None:
    now = time.time()
    for t in [k for k, v in _verify_tokens.items() if v["exp"] < now]:
        _verify_tokens.pop(t, None)


def _cleanup_reset_tokens() -> None:
    now = time.time()
    for t in [k for k, v in _reset_tokens.items() if v["exp"] < now]:
        _reset_tokens.pop(t, None)


def _json_safe_float(v):
    import math
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ─── Pydantic схемы ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or len(v) < 5:
            raise ValueError("Некорректный email")
        return v

    @field_validator("password")
    @classmethod
    def password_min_len(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Пароль минимум 6 символов")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def pwd_min(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Пароль минимум 6 символов")
        return v


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def new_password_min(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Новый пароль минимум 6 символов")
        return v


class RecordRequest(BaseModel):
    filename: str = ""
    style: str = "standard"
    out_format: str = "wav"
    before_lufs: Optional[float] = None
    after_lufs: Optional[float] = None
    target_lufs: Optional[float] = None
    duration_sec: Optional[float] = None


class ApiKeyCreate(BaseModel):
    name: str = "My API Key"


class SavedPresetCreate(BaseModel):
    name: str
    config: dict
    style: str = "standard"
    target_lufs: float = -14.0


# ─── Регистрация и вход ───────────────────────────────────────────────────────

@router.post("/api/auth/register")
def api_auth_register(body: RegisterRequest, request: Request, db=Depends(get_db)):
    """Зарегистрировать нового пользователя. P41: при REQUIRE_EMAIL_VERIFY=true — отправляет письмо."""
    from ..deps import require_feature_registration
    require_feature_registration()
    require_auth_available()
    if not check_auth_rate_limit(get_client_ip(request)):
        raise HTTPException(429, "Слишком много попыток. Подождите 1 минуту.")
    if get_user_by_email(db, body.email):
        raise HTTPException(400, "Пользователь с таким email уже существует")
    hashed = get_password_hash(body.password)
    need_verify = getattr(settings, "require_email_verify", False)
    user = create_user(db, body.email, hashed, tier="pro")

    if need_verify and DB_AVAILABLE:
        try:
            user.is_verified = False
            db.commit()
        except Exception:  # noqa: BLE001
            pass

    try:
        import asyncio as _asyncio
        if need_verify:
            _cleanup_verify_tokens()
            vtoken = secrets.token_urlsafe(32)
            _verify_tokens[vtoken] = {"email": user.email, "exp": time.time() + _VERIFY_TOKEN_TTL}
            base = str(request.base_url).rstrip("/")
            verify_url = f"{base}/verify-email?token={vtoken}"
            from ..mailer import send_email_verification
            _asyncio.get_event_loop().run_in_executor(None, send_email_verification, user.email, verify_url)
        else:
            from ..mailer import send_welcome_email
            _asyncio.get_event_loop().run_in_executor(None, send_welcome_email, user.email, user.email)
    except Exception:  # noqa: BLE001
        pass

    try:
        from ..notifier import notify_new_user
        notify_new_user(user.email, user.tier)
    except Exception:  # noqa: BLE001
        pass

    if need_verify:
        return {
            "message": "Аккаунт создан. Проверьте почту и подтвердите email для входа.",
            "email": user.email,
            "requires_verification": True,
        }

    token = create_access_token(user.id, user.email, user.tier, is_admin=bool(getattr(user, "is_admin", False)))
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "tier": user.tier,
        "is_admin": bool(getattr(user, "is_admin", False)),
    }


@router.post("/api/auth/login")
def api_auth_login(body: LoginRequest, request: Request, db=Depends(get_db)):
    """Войти. Возвращает JWT токен."""
    require_auth_available()
    if not check_auth_rate_limit(get_client_ip(request)):
        raise HTTPException(429, "Слишком много попыток входа. Подождите 1 минуту.")
    user = get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Неверный email или пароль")
    if getattr(settings, "require_email_verify", False) and not getattr(user, "is_verified", True):
        raise HTTPException(403, "Email не подтверждён. Проверьте почту.")
    if getattr(user, "is_blocked", False):
        raise HTTPException(403, "Аккаунт заблокирован. Обратитесь в поддержку.")
    user.last_login_at = time.time()
    db.commit()
    token = create_access_token(user.id, user.email, user.tier, is_admin=bool(getattr(user, "is_admin", False)))
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "tier": user.tier,
        "is_admin": bool(getattr(user, "is_admin", False)),
    }


@router.get("/api/auth/me")
def api_auth_me(user: Optional[dict] = Depends(get_current_user_optional)):
    if not user:
        raise HTTPException(401, "Не авторизован")
    return {
        "email": user.get("email"),
        "tier": user.get("tier", "free"),
        "user_id": user.get("sub"),
        "is_admin": bool(user.get("is_admin", False)),
    }


@router.post("/api/auth/logout")
def api_auth_logout():
    return {"message": "Токен удалён на стороне клиента"}


@router.get("/api/auth/verify-email")
def api_auth_verify_email(token: str, db=Depends(get_db)):
    """Подтвердить email по токену из письма. P41."""
    require_auth_available()
    _cleanup_verify_tokens()
    entry = _verify_tokens.get(token)
    if not entry or entry["exp"] < time.time():
        raise HTTPException(400, "Ссылка недействительна или истекла.")
    db_user = get_user_by_email(db, entry["email"])
    if not db_user:
        raise HTTPException(404, "Пользователь не найден")
    db_user.is_verified = True
    db.commit()
    _verify_tokens.pop(token, None)
    try:
        from ..mailer import send_welcome_email
        import asyncio as _a
        _a.get_event_loop().run_in_executor(None, send_welcome_email, db_user.email, db_user.email)
    except Exception:  # noqa: BLE001
        pass
    return {"message": "Email подтверждён! Теперь вы можете войти.", "email": db_user.email}


@router.post("/api/auth/resend-verification")
def api_auth_resend_verification(body: ForgotPasswordRequest, request: Request, db=Depends(get_db)):
    """Повторно отправить письмо верификации. P41."""
    require_auth_available()
    if not check_auth_rate_limit(get_client_ip(request)):
        raise HTTPException(429, "Слишком много попыток. Подождите 1 минуту.")
    db_user = get_user_by_email(db, body.email.strip().lower())
    if db_user and not getattr(db_user, "is_verified", True):
        _cleanup_verify_tokens()
        vtoken = secrets.token_urlsafe(32)
        _verify_tokens[vtoken] = {"email": db_user.email, "exp": time.time() + _VERIFY_TOKEN_TTL}
        base = str(request.base_url).rstrip("/")
        verify_url = f"{base}/verify-email?token={vtoken}"
        try:
            from ..mailer import send_email_verification
            import asyncio as _a
            _a.get_event_loop().run_in_executor(None, send_email_verification, db_user.email, verify_url)
        except Exception:  # noqa: BLE001
            pass
    return {"message": "Если аккаунт ожидает верификации — письмо отправлено."}


@router.get("/api/auth/profile")
def api_auth_profile(db=Depends(get_db), user: Optional[dict] = Depends(get_current_user_optional)):
    """Полная информация профиля. P31."""
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    db_user = get_user_by_email(db, user.get("email", ""))
    if not db_user:
        raise HTTPException(404, "Пользователь не найден")
    stats = get_user_stats(db, int(user["sub"]))
    return {
        "email": db_user.email,
        "tier": db_user.tier,
        "is_admin": bool(getattr(db_user, "is_admin", False)),
        "is_blocked": bool(getattr(db_user, "is_blocked", False)),
        "subscription_status": getattr(db_user, "subscription_status", "none"),
        "subscription_expires_at": getattr(db_user, "subscription_expires_at", None),
        "created_at": db_user.created_at,
        "last_login_at": getattr(db_user, "last_login_at", None),
        "stats": stats,
    }


@router.post("/api/auth/change-password")
def api_auth_change_password(
    body: ChangePasswordRequest,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    db_user = get_user_by_email(db, user["email"])
    if not db_user or not verify_password(body.old_password, db_user.hashed_password):
        raise HTTPException(400, "Неверный текущий пароль")
    db_user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    return {"message": "Пароль успешно изменён"}


@router.post("/api/auth/forgot-password")
def api_auth_forgot_password(body: ForgotPasswordRequest, request: Request, db=Depends(get_db)):
    """Запросить ссылку для сброса пароля. P35."""
    require_auth_available()
    if not check_auth_rate_limit(get_client_ip(request)):
        raise HTTPException(429, "Слишком много запросов. Подождите 1 минуту.")
    _cleanup_reset_tokens()
    db_user = get_user_by_email(db, body.email.strip().lower())
    if db_user:
        token = secrets.token_urlsafe(32)
        _reset_tokens[token] = {"email": db_user.email, "exp": time.time() + _RESET_TOKEN_TTL}
        base = str(request.base_url).rstrip("/")
        reset_url = f"{base}/reset-password?token={token}"
        try:
            from ..mailer import send_password_reset_email
            import asyncio as _a
            _a.get_event_loop().run_in_executor(None, send_password_reset_email, db_user.email, reset_url)
        except Exception:  # noqa: BLE001
            pass
    return {"message": "Если аккаунт с таким email существует, письмо со ссылкой отправлено."}


@router.post("/api/auth/reset-password")
def api_auth_reset_password(body: ResetPasswordRequest, db=Depends(get_db)):
    """Сбросить пароль по токену из email. P35."""
    require_auth_available()
    _cleanup_reset_tokens()
    entry = _reset_tokens.get(body.token)
    if not entry or entry["exp"] < time.time():
        raise HTTPException(400, "Ссылка недействительна или истекла.")
    db_user = get_user_by_email(db, entry["email"])
    if not db_user:
        raise HTTPException(404, "Пользователь не найден")
    db_user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    _reset_tokens.pop(body.token, None)
    return {"message": "Пароль успешно изменён. Войдите с новым паролем."}


# ─── История ──────────────────────────────────────────────────────────────────

@router.post("/api/auth/record")
def api_auth_record(
    body: RecordRequest,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    rec = create_mastering_record(
        db, user_id=user_id, filename=body.filename,
        style=body.style, out_format=body.out_format,
        before_lufs=body.before_lufs, after_lufs=body.after_lufs,
        target_lufs=body.target_lufs, duration_sec=body.duration_sec,
    )
    if rec is None:
        raise HTTPException(503, "База данных недоступна")
    return {"id": rec.id, "created_at": rec.created_at}


@router.get("/api/auth/history")
def api_auth_history(db=Depends(get_db), user: Optional[dict] = Depends(get_current_user_optional)):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    records = get_user_history(db, user_id, limit=30)
    stats = get_user_stats(db, user_id)
    return {
        "stats": {
            "total": stats.get("total", 0),
            "avg_lufs_change": _json_safe_float(stats.get("avg_lufs_change")),
            "top_style": stats.get("top_style"),
        },
        "records": [
            {
                "id": r.id,
                "filename": r.filename,
                "style": r.style,
                "out_format": r.out_format,
                "before_lufs": _json_safe_float(r.before_lufs),
                "after_lufs": _json_safe_float(r.after_lufs),
                "target_lufs": _json_safe_float(r.target_lufs),
                "duration_sec": _json_safe_float(r.duration_sec),
                "created_at": _json_safe_float(r.created_at),
            }
            for r in records
        ],
    }


@router.delete("/api/auth/history/{record_id}")
def api_auth_history_delete(
    record_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if not delete_mastering_record(db, record_id, int(user["sub"])):
        raise HTTPException(404, "Запись не найдена")
    return {"deleted": record_id}


@router.get("/api/auth/history/export.csv")
def api_auth_history_export_csv(
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """Скачать историю мастерингов как CSV-файл. P42."""
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    records = get_user_history(db, int(user["sub"]), limit=10000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "filename", "style", "out_format",
                "before_lufs", "after_lufs", "target_lufs", "duration_sec", "date"])
    for r in records:
        w.writerow([
            r.id, r.filename or "", r.style or "", r.out_format or "",
            f"{r.before_lufs:.2f}" if r.before_lufs is not None else "",
            f"{r.after_lufs:.2f}" if r.after_lufs is not None else "",
            f"{r.target_lufs:.2f}" if r.target_lufs is not None else "",
            f"{r.duration_sec:.1f}" if r.duration_sec is not None else "",
            datetime.datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d %H:%M") if r.created_at else "",
        ])
    content = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=mastering_history.csv"},
    )


# ─── API ключи (P52) ──────────────────────────────────────────────────────────

@router.get("/api/auth/api-keys")
def api_keys_list(db=Depends(get_db), user: Optional[dict] = Depends(get_current_user_optional)):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if user.get("tier", "free") not in ("pro", "studio") and not user.get("is_admin"):
        raise HTTPException(403, "API-ключи доступны только для тарифов Pro и Studio")
    keys = get_api_keys_for_user(db, int(user["sub"]))
    return {
        "keys": [
            {"id": k.id, "name": k.name, "prefix": k.key_prefix + "…",
             "is_active": k.is_active, "created_at": k.created_at, "last_used_at": k.last_used_at}
            for k in keys
        ]
    }


@router.post("/api/auth/api-keys", status_code=201)
def api_keys_create(
    body: ApiKeyCreate,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if user.get("tier", "free") not in ("pro", "studio") and not user.get("is_admin"):
        raise HTTPException(403, "API-ключи доступны только для тарифов Pro и Studio")
    existing = [k for k in get_api_keys_for_user(db, int(user["sub"])) if k.is_active]
    if len(existing) >= 10:
        raise HTTPException(400, "Достигнут лимит: не более 10 активных API-ключей")
    key_obj, raw_key = create_api_key(db, int(user["sub"]), body.name.strip() or "My API Key")
    return {
        "id": key_obj.id, "name": key_obj.name, "prefix": key_obj.key_prefix + "…",
        "key": raw_key, "created_at": key_obj.created_at,
        "warning": "Сохраните ключ сейчас — он больше не будет показан.",
    }


@router.delete("/api/auth/api-keys/{key_id}")
def api_keys_revoke(
    key_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if not revoke_api_key(db, key_id, int(user["sub"])):
        raise HTTPException(404, "Ключ не найден")
    return {"revoked": True, "key_id": key_id}


# ─── Сохранённые пресеты (P10) ────────────────────────────────────────────────

@router.get("/api/auth/presets")
def api_auth_presets_list(db=Depends(get_db), user: Optional[dict] = Depends(get_current_user_optional)):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    presets = get_user_presets(db, int(user["sub"]), limit=50)
    return {
        "presets": [
            {
                "id": p.id, "name": p.name,
                "config": json.loads(p.config) if p.config else {},
                "style": p.style, "target_lufs": p.target_lufs, "created_at": p.created_at,
            }
            for p in presets
        ]
    }


@router.post("/api/auth/presets")
def api_auth_presets_create(
    body: SavedPresetCreate,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    config_str = json.dumps(body.config) if isinstance(body.config, dict) else "{}"
    preset = create_saved_preset(
        db, int(user["sub"]), body.name, config_str,
        style=body.style, target_lufs=body.target_lufs,
    )
    return {"id": preset.id, "name": preset.name, "created_at": preset.created_at}


@router.get("/api/auth/presets/{preset_id}")
def api_auth_presets_get(
    preset_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    preset = get_saved_preset_by_id(db, preset_id, int(user["sub"]))
    if not preset:
        raise HTTPException(404, "Пресет не найден")
    return {
        "id": preset.id, "name": preset.name,
        "config": json.loads(preset.config) if preset.config else {},
        "style": preset.style, "target_lufs": preset.target_lufs, "created_at": preset.created_at,
    }


@router.delete("/api/auth/presets/{preset_id}")
def api_auth_presets_delete(
    preset_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if not delete_saved_preset(db, preset_id, int(user["sub"])):
        raise HTTPException(404, "Пресет не найден")
    return {"deleted": preset_id}
