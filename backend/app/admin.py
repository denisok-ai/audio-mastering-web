"""Admin API router — /api/admin/* endpoints (P19).

Все эндпоинты защищены зависимостью _get_current_admin (JWT + is_admin=True).
"""
import csv
import io
import json
import os
import time
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .auth import extract_bearer_token, decode_token
from fastapi import Request as _Request

from .database import (
    DB_AVAILABLE,
    AiUsageLog,
    AuditLog,
    EmailCampaign,
    MasteringJobEvent,
    MasteringRecord,
    NewsPost,
    PromptTemplate,
    SavedPreset,
    SessionLocal,
    Transaction,
    User,
    create_email_campaign,
    create_news_post,
    create_prompt_version,
    create_transaction,
    delete_news_post,
    get_active_prompt_body,
    get_audit_logs,
    count_email_campaigns,
    count_news_posts,
    count_transactions,
    get_email_campaign_by_id,
    get_email_campaigns,
    get_prompt_history,
    get_news_post_by_id,
    get_news_posts,
    get_transactions,
    reset_prompt_to_builtin,
    set_active_prompt,
    update_email_campaign,
    update_news_post,
    write_audit_log,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ─── Auth dependency ──────────────────────────────────────────────────────────

def _get_current_admin(authorization: Optional[str] = Header(None)) -> dict:
    """Требует авторизации + is_admin=True в JWT."""
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(401, "Требуется авторизация администратора")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Недействительный или просроченный токен")
    if not payload.get("is_admin"):
        raise HTTPException(403, "Доступ запрещён: требуются права администратора")
    return payload


def _get_db():
    if not DB_AVAILABLE or SessionLocal is None:
        raise HTTPException(503, "База данных недоступна")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class UserPatch(BaseModel):
    tier: Optional[str] = None
    is_blocked: Optional[bool] = None
    is_admin: Optional[bool] = None


class SubscriptionSet(BaseModel):
    tier: str
    expires_at: Optional[float] = None   # unix timestamp; None = без срока
    amount: float = 0.0
    description: str = ""


class ManualTransaction(BaseModel):
    user_id: int
    amount: float
    currency: str = "RUB"
    tier: str
    description: str = ""
    status: str = "succeeded"


class NewsCreate(BaseModel):
    title: str
    body: str
    is_published: bool = False


class NewsUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    is_published: Optional[bool] = None


class CampaignCreate(BaseModel):
    subject: str
    body_html: str
    body_text: str = ""
    target_tier: Optional[str] = None   # None = все тарифы


class SettingsAppUpdate(BaseModel):
    max_upload_mb: Optional[int] = None
    allowed_extensions: Optional[List[str]] = None
    temp_dir: Optional[str] = None
    default_target_lufs: Optional[float] = None
    jobs_done_ttl_seconds: Optional[int] = None
    debug_mode: Optional[bool] = None
    require_email_verify: Optional[bool] = None
    global_rate_limit: Optional[int] = None
    cors_origins: Optional[str] = None
    feature_ai_enabled: Optional[bool] = None
    feature_batch_enabled: Optional[bool] = None
    feature_registration_enabled: Optional[bool] = None
    maintenance_mode: Optional[bool] = None
    notify_email_on_register: Optional[bool] = None
    notify_telegram_on_payment: Optional[bool] = None
    default_locale: Optional[str] = None


class SettingsSmtpUpdate(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None  # пусто = не менять
    from_: Optional[str] = None
    use_tls: Optional[bool] = None


class SettingsYookassaUpdate(BaseModel):
    shop_id: Optional[str] = None
    secret_key: Optional[str] = None  # пусто = не менять
    return_url: Optional[str] = None
    webhook_ip_whitelist: Optional[str] = None


class SettingsTelegramUpdate(BaseModel):
    bot_token: Optional[str] = None  # пусто = не менять
    chat_id: Optional[str] = None


class SettingsLlmUpdate(BaseModel):
    backend: Optional[str] = None
    openai_api_key: Optional[str] = None  # пусто = не менять
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: Optional[str] = None
    deepseek_model: Optional[str] = None
    ai_limit_free: Optional[int] = None
    ai_limit_pro: Optional[int] = None
    ai_limit_studio: Optional[int] = None
    # Защита от LLM-инъекций
    llm_guard_enabled: Optional[bool] = None
    llm_guard_max_length_chat: Optional[int] = None
    llm_guard_max_length_nl: Optional[int] = None
    llm_guard_forbidden_substrings: Optional[list[str]] = None  # JSON array
    llm_guard_forbidden_regex: Optional[str] = None
    llm_guard_truncate_on_long: Optional[bool] = None
    llm_guard_block_role_system: Optional[bool] = None


class SettingsUpdate(BaseModel):
    app: Optional[SettingsAppUpdate] = None
    smtp: Optional[SettingsSmtpUpdate] = None
    yookassa: Optional[SettingsYookassaUpdate] = None
    telegram: Optional[SettingsTelegramUpdate] = None
    llm: Optional[SettingsLlmUpdate] = None


class PromptCreate(BaseModel):
    slug: str  # recommend | report | nl_config | chat
    body: str
    save_as_template_name: Optional[str] = None


class PromptActivate(BaseModel):
    version_id: int


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _user_to_dict(u: "User") -> dict:
    return {
        "id": u.id,
        "email": u.email,
        "tier": u.tier,
        "is_admin": bool(getattr(u, "is_admin", False)),
        "is_blocked": bool(getattr(u, "is_blocked", False)),
        "subscription_status": getattr(u, "subscription_status", "none"),
        "subscription_expires_at": getattr(u, "subscription_expires_at", None),
        "created_at": u.created_at,
        "last_login_at": getattr(u, "last_login_at", None),
    }


def _tx_to_dict(t: "Transaction") -> dict:
    return {
        "id": t.id,
        "user_id": t.user_id,
        "amount": t.amount,
        "currency": t.currency,
        "tier": t.tier,
        "payment_system": t.payment_system,
        "external_id": t.external_id,
        "status": t.status,
        "description": t.description,
        "created_at": t.created_at,
    }


def _news_to_dict(p: "NewsPost") -> dict:
    return {
        "id": p.id,
        "title": p.title,
        "body": p.body,
        "author_id": p.author_id,
        "is_published": p.is_published,
        "published_at": p.published_at,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
    }


def _campaign_to_dict(c: "EmailCampaign") -> dict:
    return {
        "id": c.id,
        "subject": c.subject,
        "body_html": c.body_html,
        "body_text": c.body_text,
        "target_tier": c.target_tier,
        "status": c.status,
        "total_recipients": c.total_recipients,
        "sent_count": c.sent_count,
        "created_at": c.created_at,
        "sent_at": c.sent_at,
    }


# ─── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats")
def admin_stats(
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Метрики для Dashboard: пользователи, транзакции, мастеринги. P43: расширенная аналитика."""
    import datetime as _dt

    users = db.query(User).all()
    tier_counts: dict = {}
    for u in users:
        tier_counts[u.tier] = tier_counts.get(u.tier, 0) + 1

    total_users = len(users)
    blocked_count = sum(1 for u in users if getattr(u, "is_blocked", False))
    unverified_count = sum(1 for u in users if not getattr(u, "is_verified", True))

    # Новые пользователи за 30 и 7 дней
    now = time.time()
    month_start = now - 30 * 86400
    week_start = now - 7 * 86400
    new_users_month = sum(1 for u in users if u.created_at and u.created_at >= month_start)
    new_users_week = sum(1 for u in users if u.created_at and u.created_at >= week_start)

    # Новые пользователи по дням (последние 7 дней) — для sparkline
    today_date = _dt.date.today()
    new_users_by_day = []
    for i in range(6, -1, -1):
        day = today_date - _dt.timedelta(days=i)
        day_ts_start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        day_ts_end = _dt.datetime.combine(day, _dt.time.max).timestamp()
        count = sum(1 for u in users if u.created_at and day_ts_start <= u.created_at <= day_ts_end)
        new_users_by_day.append({"date": day.isoformat(), "count": count})

    # Транзакции (выручка)
    all_txs = db.query(Transaction).filter(Transaction.status == "succeeded").all()
    revenue_total = round(sum(getattr(t, "amount", 0) or 0 for t in all_txs), 2)
    txs_month = [t for t in all_txs if t.created_at and t.created_at >= month_start]
    revenue_month = round(sum(getattr(t, "amount", 0) or 0 for t in txs_month), 2)
    tx_count_month = len(txs_month)

    # Выручка по дням (7 дней) — для sparkline
    revenue_by_day = []
    for i in range(6, -1, -1):
        day = today_date - _dt.timedelta(days=i)
        day_ts_start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        day_ts_end = _dt.datetime.combine(day, _dt.time.max).timestamp()
        amount = sum(
            getattr(t, "amount", 0) or 0
            for t in all_txs
            if t.created_at and day_ts_start <= t.created_at <= day_ts_end
        )
        revenue_by_day.append({"date": day.isoformat(), "amount": round(amount, 2)})

    # Мастеринги
    day_start = now - 86400
    masters_today = db.query(MasteringRecord).filter(MasteringRecord.created_at >= day_start).count()
    masters_month = db.query(MasteringRecord).filter(MasteringRecord.created_at >= month_start).count()
    masters_total = db.query(MasteringRecord).count()

    # Мастеринги по дням (7 дней) — для sparkline
    masters_by_day = []
    for i in range(6, -1, -1):
        day = today_date - _dt.timedelta(days=i)
        day_ts_start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        day_ts_end = _dt.datetime.combine(day, _dt.time.max).timestamp()
        count = db.query(MasteringRecord).filter(
            MasteringRecord.created_at >= day_ts_start,
            MasteringRecord.created_at <= day_ts_end,
        ).count()
        masters_by_day.append({"date": day.isoformat(), "count": count})

    # Новости
    news_total = db.query(NewsPost).count()
    news_published = db.query(NewsPost).filter(NewsPost.is_published == True).count()  # noqa: E712

    # Активные подписки
    active_subs = sum(
        1 for u in users
        if getattr(u, "subscription_status", "none") == "active"
        and getattr(u, "subscription_expires_at", None)
        and u.subscription_expires_at > now
    )

    return {
        "users": {
            "total": total_users,
            "blocked": blocked_count,
            "unverified": unverified_count,
            "by_tier": tier_counts,
            "new_month": new_users_month,
            "new_week": new_users_week,
            "by_day": new_users_by_day,
            "active_subscriptions": active_subs,
        },
        "revenue": {
            "total_rub": revenue_total,
            "month_rub": revenue_month,
            "transactions_month": tx_count_month,
            "by_day": revenue_by_day,
        },
        "masterings": {
            "today": masters_today,
            "month": masters_month,
            "total": masters_total,
            "by_day": masters_by_day,
        },
        "news": {
            "total": news_total,
            "published": news_published,
        },
    }


# ─── User management ─────────────────────────────────────────────────────────

@router.get("/users")
def admin_list_users(
    search: str = "",
    tier: str = "",
    blocked: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Список пользователей с поиском и фильтрацией."""
    q = db.query(User)
    if search:
        q = q.filter(User.email.ilike(f"%{search}%"))
    if tier:
        q = q.filter(User.tier == tier)
    if blocked is not None:
        q = q.filter(User.is_blocked == blocked)
    total = q.count()
    users = q.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "users": [_user_to_dict(u) for u in users],
    }


@router.get("/users/{user_id}")
def admin_get_user(
    user_id: int,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Детали пользователя + последние записи мастеринга."""
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    records = (
        db.query(MasteringRecord)
        .filter(MasteringRecord.user_id == user_id)
        .order_by(MasteringRecord.created_at.desc())
        .limit(20)
        .all()
    )
    presets = db.query(SavedPreset).filter(SavedPreset.user_id == user_id).count()
    txs = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.created_at.desc())
        .limit(10)
        .all()
    )
    return {
        **_user_to_dict(u),
        "mastering_count": len(records),
        "preset_count": presets,
        "recent_records": [
            {
                "id": r.id,
                "filename": r.filename,
                "style": r.style,
                "before_lufs": r.before_lufs,
                "after_lufs": r.after_lufs,
                "created_at": r.created_at,
            }
            for r in records
        ],
        "recent_transactions": [_tx_to_dict(t) for t in txs],
    }


@router.patch("/users/{user_id}")
def admin_patch_user(
    user_id: int,
    body: UserPatch,
    request: _Request,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Изменить tier, is_blocked или is_admin пользователя."""
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    changes = []
    if body.tier is not None:
        if body.tier not in ("free", "pro", "studio"):
            raise HTTPException(400, "tier должен быть free, pro или studio")
        changes.append(f"tier: {u.tier} → {body.tier}")
        u.tier = body.tier
    if body.is_blocked is not None:
        changes.append(f"blocked: {u.is_blocked} → {body.is_blocked}")
        u.is_blocked = body.is_blocked
    if body.is_admin is not None:
        changes.append(f"is_admin: {u.is_admin} → {body.is_admin}")
        u.is_admin = body.is_admin
    db.commit()
    db.refresh(u)
    # P55: запись в журнал
    write_audit_log(
        db, int(admin.get("sub", 0) or 0), admin.get("email", ""),
        action="patch_user", target_type="user", target_id=user_id,
        details="; ".join(changes) or "no changes",
        ip=request.client.host if request.client else "",
    )
    return _user_to_dict(u)


@router.delete("/users/{user_id}")
def admin_delete_user(
    user_id: int,
    request: _Request,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Удалить пользователя и все его данные (CASCADE)."""
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    if str(user_id) == str(admin.get("sub")):
        raise HTTPException(400, "Нельзя удалить собственную учётную запись через API")
    user_email = u.email
    db.delete(u)
    db.commit()
    # P55: запись в журнал
    write_audit_log(
        db, int(admin.get("sub", 0) or 0), admin.get("email", ""),
        action="delete_user", target_type="user", target_id=user_id,
        details=f"email: {user_email}",
        ip=request.client.host if request.client else "",
    )
    return {"deleted": True, "user_id": user_id}


class BulkActionRequest(BaseModel):
    user_ids: List[int]
    action: str          # "block" | "unblock" | "delete" | "set_tier"
    tier: Optional[str] = None   # обязательно для set_tier


@router.post("/users/bulk-action")
def admin_bulk_action(
    body: BulkActionRequest,
    request: _Request,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Массовое действие над пользователями. P48.

    action=block    — заблокировать выбранных
    action=unblock  — разблокировать
    action=delete   — удалить (нельзя удалить самого себя)
    action=set_tier — изменить тариф (требует tier)
    """
    if not body.user_ids:
        raise HTTPException(400, "user_ids не может быть пустым")
    if body.action not in ("block", "unblock", "delete", "set_tier"):
        raise HTTPException(400, "action: block | unblock | delete | set_tier")
    if body.action == "set_tier":
        if body.tier not in ("free", "pro", "studio"):
            raise HTTPException(400, "tier должен быть free, pro или studio")

    admin_id = str(admin.get("sub"))
    affected = 0
    skipped  = 0
    errors: list = []

    for uid in body.user_ids:
        if body.action == "delete" and str(uid) == admin_id:
            skipped += 1
            continue
        u = db.query(User).filter(User.id == uid).first()
        if not u:
            skipped += 1
            continue
        try:
            if body.action == "block":
                u.is_blocked = True
            elif body.action == "unblock":
                u.is_blocked = False
            elif body.action == "set_tier":
                u.tier = body.tier
            elif body.action == "delete":
                db.delete(u)
            affected += 1
        except Exception as e:
            errors.append({"user_id": uid, "error": str(e)[:100]})
            skipped += 1

    db.commit()
    # P55: запись в журнал
    write_audit_log(
        db, int(admin.get("sub", 0) or 0), admin.get("email", ""),
        action=f"bulk_{body.action}", target_type="user",
        details=f"ids={body.user_ids[:20]}; affected={affected}; tier={body.tier or ''}",
        ip=request.client.host if request.client else "",
    )
    return {
        "action": body.action,
        "affected": affected,
        "skipped": skipped,
        "errors": errors,
    }


@router.post("/users/{user_id}/subscription")
def admin_set_subscription(
    user_id: int,
    body: SubscriptionSet,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Назначить подписку пользователю вручную и создать транзакцию."""
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    if body.tier not in ("free", "pro", "studio"):
        raise HTTPException(400, "tier должен быть free, pro или studio")
    u.tier = body.tier
    u.subscription_expires_at = body.expires_at
    u.subscription_status = "active" if body.tier != "free" else "none"
    db.commit()

    tx = create_transaction(
        db,
        user_id=user_id,
        amount=body.amount,
        tier=body.tier,
        payment_system="manual",
        status="succeeded",
        description=body.description or f"Ручное назначение тарифа {body.tier}",
    )
    db.refresh(u)
    # Письмо об активации подписки (P22)
    try:
        from .mailer import send_subscription_activated_email
        import asyncio as _asyncio
        _asyncio.get_event_loop().run_in_executor(
            None, send_subscription_activated_email, u.email, body.tier, body.expires_at
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "user": _user_to_dict(u),
        "transaction": _tx_to_dict(tx) if tx else None,
    }


# ─── Transactions ─────────────────────────────────────────────────────────────

@router.get("/transactions")
def admin_list_transactions(
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Список транзакций с фильтрацией и пагинацией."""
    txs = get_transactions(db, user_id=user_id, status=status, limit=limit, offset=offset)
    total = count_transactions(db, user_id=user_id, status=status)
    return {"total": total, "limit": limit, "offset": offset, "transactions": [_tx_to_dict(t) for t in txs]}


@router.post("/transactions")
def admin_create_transaction(
    body: ManualTransaction,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Добавить ручную транзакцию."""
    u = db.query(User).filter(User.id == body.user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    tx = create_transaction(
        db,
        user_id=body.user_id,
        amount=body.amount,
        tier=body.tier,
        payment_system="manual",
        currency=body.currency,
        status=body.status,
        description=body.description,
    )
    return _tx_to_dict(tx)


# ─── News ─────────────────────────────────────────────────────────────────────

@router.get("/news")
def admin_list_news(
    published_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Список всех постов (включая черновики) с пагинацией."""
    posts = get_news_posts(db, published_only=published_only, limit=limit, offset=offset)
    total = count_news_posts(db, published_only=published_only)
    return {"total": total, "limit": limit, "offset": offset, "posts": [_news_to_dict(p) for p in posts]}


@router.post("/news")
def admin_create_news(
    body: NewsCreate,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Создать новость или объявление."""
    author_id = int(admin.get("sub", 0)) or None
    post = create_news_post(db, title=body.title, body=body.body,
                            author_id=author_id, is_published=body.is_published)
    return _news_to_dict(post)


@router.put("/news/{post_id}")
def admin_update_news(
    post_id: int,
    body: NewsUpdate,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Обновить пост (текст, статус публикации)."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Нет полей для обновления")
    post = update_news_post(db, post_id, **updates)
    if not post:
        raise HTTPException(404, "Пост не найден")
    return _news_to_dict(post)


@router.delete("/news/{post_id}")
def admin_delete_news(
    post_id: int,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Удалить пост."""
    ok = delete_news_post(db, post_id)
    if not ok:
        raise HTTPException(404, "Пост не найден")
    return {"deleted": True, "post_id": post_id}


# ─── Email campaigns ─────────────────────────────────────────────────────────

@router.get("/campaigns")
def admin_list_campaigns(
    limit: int = 20,
    offset: int = 0,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Список email-кампаний с пагинацией."""
    camps = get_email_campaigns(db, limit=limit, offset=offset)
    total = count_email_campaigns(db)
    return {"total": total, "limit": limit, "offset": offset, "campaigns": [_campaign_to_dict(c) for c in camps]}


@router.post("/campaigns")
def admin_create_campaign(
    body: CampaignCreate,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Создать новую кампанию (статус draft)."""
    camp = create_email_campaign(
        db,
        subject=body.subject,
        body_html=body.body_html,
        body_text=body.body_text,
        target_tier=body.target_tier,
    )
    return _campaign_to_dict(camp)


@router.post("/campaigns/{campaign_id}/send")
async def admin_send_campaign(
    campaign_id: int,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Запустить рассылку кампании в фоне (SMTP)."""
    from fastapi import BackgroundTasks
    camp = get_email_campaign_by_id(db, campaign_id)
    if not camp:
        raise HTTPException(404, "Кампания не найдена")
    if camp.status == "sending":
        raise HTTPException(400, "Кампания уже отправляется")
    if camp.status == "sent":
        raise HTTPException(400, "Кампания уже отправлена")

    # Подсчитываем получателей
    q = db.query(User).filter(User.is_blocked == False)  # noqa: E712
    if camp.target_tier:
        q = q.filter(User.tier == camp.target_tier)
    recipients = [u.email for u in q.all()]
    total = len(recipients)

    update_email_campaign(db, campaign_id, status="sending", total_recipients=total, sent_count=0)

    # Фоновая задача отправки
    import asyncio

    async def _do_send():
        from .mailer import send_email
        from .config import settings as _cfg
        sent = 0
        failed = 0
        for email_addr in recipients:
            try:
                await asyncio.to_thread(
                    send_email,
                    to=email_addr,
                    subject=camp.subject,
                    html=camp.body_html,
                    text=camp.body_text,
                )
                sent += 1
            except Exception:  # noqa: BLE001
                failed += 1
            # небольшая пауза между письмами
            await asyncio.sleep(0.1)
        new_db = SessionLocal()
        try:
            update_email_campaign(new_db, campaign_id,
                                  status="sent" if failed == 0 else "failed",
                                  sent_count=sent,
                                  sent_at=time.time())
        finally:
            new_db.close()

    asyncio.create_task(_do_send())

    return {
        "campaign_id": campaign_id,
        "status": "sending",
        "total_recipients": total,
        "message": f"Рассылка запущена: {total} получателей",
    }


@router.get("/campaigns/{campaign_id}/stats")
def admin_campaign_stats(
    campaign_id: int,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Статус и статистика кампании."""
    camp = get_email_campaign_by_id(db, campaign_id)
    if not camp:
        raise HTTPException(404, "Кампания не найдена")
    return _campaign_to_dict(camp)


# ─── Settings endpoint (P29) ────────────────────────────────────────────────

@router.get("/audit")
def admin_get_audit_log(
    limit: int = 50,
    offset: int = 0,
    admin_id: Optional[int] = None,
    action: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Журнал административных действий. P55. Фильтры: action, date_from, date_to (YYYY-MM-DD)."""
    ts_from = None
    ts_to = None
    if date_from:
        try:
            from datetime import datetime
            dt = datetime.strptime(date_from.strip()[:10], "%Y-%m-%d")
            ts_from = dt.timestamp()
        except (ValueError, TypeError):
            pass
    if date_to:
        try:
            from datetime import datetime, timedelta
            dt = datetime.strptime(date_to.strip()[:10], "%Y-%m-%d") + timedelta(days=1)
            ts_to = dt.timestamp()
        except (ValueError, TypeError):
            pass
    logs, total = get_audit_logs(
        db, limit=limit, offset=offset, admin_id=admin_id,
        action=action, ts_from=ts_from, ts_to=ts_to,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "logs": [
            {
                "id": lg.id,
                "admin_email": lg.admin_email,
                "action": lg.action,
                "target_type": lg.target_type,
                "target_id": lg.target_id,
                "details": lg.details,
                "ip": lg.ip,
                "created_at": lg.created_at,
            }
            for lg in logs
        ],
    }


@router.post("/notifications/test-email")
def admin_test_email(admin: dict = Depends(_get_current_admin)):
    """Отправить тестовое письмо на email администратора. P53."""
    from .config import settings as cfg
    if not cfg.smtp_host or not cfg.smtp_user:
        raise HTTPException(400, "SMTP не настроен: задайте MAGIC_MASTER_SMTP_HOST и MAGIC_MASTER_SMTP_USER")
    admin_email = str(admin.get("email", cfg.admin_email or cfg.smtp_user))
    try:
        from .mailer import send_email
        send_email(
            to=admin_email,
            subject="✅ Тест SMTP — Magic Master",
            html="<h2>Тест SMTP работает!</h2><p>Это тестовое письмо от Magic Master.</p>",
            text="SMTP работает. Это тестовое письмо от Magic Master.",
        )
        return {"ok": True, "message": f"Письмо отправлено на {admin_email}"}
    except Exception as e:
        raise HTTPException(500, f"Ошибка отправки: {e}")


@router.post("/notifications/test-telegram")
def admin_test_telegram(admin: dict = Depends(_get_current_admin)):
    """Отправить тестовое Telegram-сообщение. P51."""
    try:
        from .notifier import notify, _is_configured
        if not _is_configured():
            raise HTTPException(400, "Telegram не настроен: задайте MAGIC_MASTER_TELEGRAM_BOT_TOKEN и MAGIC_MASTER_TELEGRAM_ADMIN_CHAT_ID")
        notify("✅ <b>Тест — Magic Master</b>\nTelegram-уведомления работают корректно.")
        return {"ok": True, "message": "Сообщение отправлено"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка: {e}")


def _effective_setting(key: str, default=None):
    """Эффективное значение настройки: из БД или config."""
    from . import settings_store
    v = settings_store.get_setting(key)
    return default if v is None else v


@router.get("/settings")
def admin_settings(admin: dict = Depends(_get_current_admin)):
    """Текущая конфигурация сервиса (редактируемые из БД, пароли/ключи скрыты)."""
    from .config import settings as cfg
    from .version import __version__ as app_version, __build_date__ as app_build_date

    def _mask(val: str) -> str:
        if not val:
            return ""
        return val[:2] + "***" + val[-2:] if len(val) > 6 else "***"

    def _str(k, d=""):
        v = _effective_setting(k)
        return (v or d) if v is not None else d

    allowed = _effective_setting("allowed_extensions")
    if allowed is None:
        allowed = list(getattr(cfg, "allowed_extensions", ["wav", "mp3", "flac"]))
    elif isinstance(allowed, set):
        allowed = list(allowed)

    return {
        "app": {
            "version": app_version,
            "build_date": app_build_date,
            "jwt_secret_set": bool(os.environ.get("MAGIC_MASTER_JWT_SECRET")),
            "allowed_extensions": allowed,
            "max_upload_mb": _effective_setting("max_upload_mb", 100),
            "jobs_done_ttl_seconds": _effective_setting("jobs_done_ttl_seconds", 3600),
            "debug_mode": _effective_setting("debug_mode", False),
            "default_target_lufs": _effective_setting("default_target_lufs", -14.0),
            "require_email_verify": _effective_setting("require_email_verify", False),
            "global_rate_limit": _effective_setting("global_rate_limit", 300),
            "cors_origins": (_str("cors_origins") or "*").strip() or "*",
            "temp_dir": _str("temp_dir", "/tmp/masterflow"),
            "feature_ai_enabled": _effective_setting("feature_ai_enabled"),
            "feature_batch_enabled": _effective_setting("feature_batch_enabled"),
            "feature_registration_enabled": _effective_setting("feature_registration_enabled"),
            "maintenance_mode": _effective_setting("maintenance_mode", False),
            "notify_email_on_register": _effective_setting("notify_email_on_register", False),
            "notify_telegram_on_payment": _effective_setting("notify_telegram_on_payment", False),
            "default_locale": _str("default_locale", "ru"),
        },
        "smtp": {
            "host": _str("smtp_host"),
            "port": _effective_setting("smtp_port", 587),
            "user": _str("smtp_user"),
            "password": _mask(_str("smtp_password")),
            "from": _str("smtp_from"),
            "use_tls": _effective_setting("smtp_use_tls", True),
            "configured": bool(_str("smtp_host") and _str("smtp_user")),
        },
        "yookassa": {
            "shop_id": _str("yookassa_shop_id"),
            "secret_key": _mask(_str("yookassa_secret_key")),
            "return_url": _str("yookassa_return_url", "http://localhost:8000/pricing"),
            "webhook_ip_whitelist": _str("yookassa_webhook_ip_whitelist"),
            "configured": bool(_str("yookassa_shop_id") and _str("yookassa_secret_key")),
        },
        "telegram": {
            "configured": bool(_str("telegram_bot_token") and _str("telegram_admin_chat_id")),
            "bot_token": _mask(_str("telegram_bot_token")),
            "chat_id": (_str("telegram_admin_chat_id"))[:8] + "…" if _str("telegram_admin_chat_id") else "—",
        },
        "admin_email": cfg.admin_email or "",
        "admin_password_set": bool((getattr(cfg, "admin_password", "") or "").strip()),
        "llm": _admin_llm_settings(_mask),
    }


def _apply_settings_update(body: SettingsUpdate, admin_id: Optional[int]) -> None:
    """Записывает переданные секции настроек в system_settings."""
    from . import settings_store
    if body.app:
        for k, v in body.app.model_dump(exclude_none=True).items():
            if k == "allowed_extensions" and v is not None:
                settings_store.set_setting("allowed_extensions", v, admin_id)
            else:
                settings_store.set_setting(k, v, admin_id)
    if body.smtp:
        for k, v in body.smtp.model_dump(exclude_none=True).items():
            key = "smtp_from" if k == "from_" else f"smtp_{k}"
            settings_store.set_setting(key, v, admin_id)
    if body.yookassa:
        key_map = {"shop_id": "yookassa_shop_id", "secret_key": "yookassa_secret_key", "return_url": "yookassa_return_url", "webhook_ip_whitelist": "yookassa_webhook_ip_whitelist"}
        for k, v in body.yookassa.model_dump(exclude_none=True).items():
            settings_store.set_setting(key_map.get(k, k), v, admin_id)
    if body.telegram:
        for k, v in body.telegram.model_dump(exclude_none=True).items():
            key = "telegram_bot_token" if k == "bot_token" else "telegram_admin_chat_id"
            settings_store.set_setting(key, v, admin_id)
    if body.llm:
        key_map = {"backend": "ai_backend", "openai_api_key": "openai_api_key", "anthropic_api_key": "anthropic_api_key", "deepseek_api_key": "deepseek_api_key", "deepseek_base_url": "deepseek_base_url", "deepseek_model": "deepseek_model", "ai_limit_free": "ai_limit_free", "ai_limit_pro": "ai_limit_pro", "ai_limit_studio": "ai_limit_studio", "llm_guard_enabled": "llm_guard_enabled", "llm_guard_max_length_chat": "llm_guard_max_length_chat", "llm_guard_max_length_nl": "llm_guard_max_length_nl", "llm_guard_forbidden_substrings": "llm_guard_forbidden_substrings", "llm_guard_forbidden_regex": "llm_guard_forbidden_regex", "llm_guard_truncate_on_long": "llm_guard_truncate_on_long", "llm_guard_block_role_system": "llm_guard_block_role_system"}
        for k, v in body.llm.model_dump(exclude_none=True).items():
            settings_store.set_setting(key_map.get(k, k), v, admin_id)


@router.patch("/settings")
def admin_settings_patch(
    body: SettingsUpdate,
    request: Request,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Обновить настройки сервиса (запись в system_settings). Секреты: пустая строка = не менять."""
    admin_id = admin.get("sub")
    try:
        admin_id_int = int(admin_id) if admin_id is not None else None
    except (TypeError, ValueError):
        admin_id_int = None
    _apply_settings_update(body, admin_id_int)
    write_audit_log(
        db,
        admin_id_int,
        admin.get("email", "") or "",
        "settings_update",
        "settings",
        None,
        details=json.dumps({"sections": [s for s in ["app", "smtp", "yookassa", "telegram", "llm"] if getattr(body, s) is not None]}),
        ip=request.client.host if request.client else "",
    )
    return {"ok": True, "message": "Настройки сохранены. Часть параметров (например CORS) применится после перезапуска."}


# ─── Prompts (LLM) ────────────────────────────────────────────────────────────

_PROMPT_SLUGS = ("recommend", "report", "nl_config", "chat")


@router.get("/prompts")
def admin_prompts_list(admin: dict = Depends(_get_current_admin), db=Depends(_get_db)):
    """Список промптов по slug: активный текст, версия, последние версии."""
    from . import ai as ai_module
    result = {}
    effective_prompts = ai_module.get_effective_prompts() or {}
    for slug in _PROMPT_SLUGS:
        active_body = get_active_prompt_body(db, slug)
        history = get_prompt_history(db, slug, limit=10)
        effective = effective_prompts.get(slug) or ""
        result[slug] = {
            "active_body": active_body,
            "effective_body": effective,
            "effective_preview": (effective[:200] + "…") if len(effective) > 200 else effective,
            "history": [
                {"id": h.id, "version": h.version, "created_at": h.created_at, "created_by": h.created_by, "preview": (h.body or "")[:120] + ("…" if len(h.body or "") > 120 else "")}
                for h in history
            ],
        }
    return result


@router.post("/prompts")
def admin_prompts_create(body: PromptCreate, admin: dict = Depends(_get_current_admin), db=Depends(_get_db)):
    """Создать новую версию промпта и сделать её активной."""
    if body.slug not in _PROMPT_SLUGS:
        raise HTTPException(400, "Неверный slug")
    admin_id = None
    try:
        admin_id = int(admin.get("sub"))
    except (TypeError, ValueError):
        pass
    row = create_prompt_version(db, body.slug, body.body.strip(), admin_id, body.save_as_template_name or "")
    if not row:
        raise HTTPException(500, "Не удалось сохранить промпт")
    write_audit_log(db, admin_id, admin.get("email", "") or "", "prompt_update", "prompt", row.id, details=body.slug, ip="")
    return {"ok": True, "id": row.id, "version": row.version}


@router.get("/prompts/{slug}/history")
def admin_prompts_history(slug: str, admin: dict = Depends(_get_current_admin), db=Depends(_get_db)):
    """История версий промпта."""
    if slug not in _PROMPT_SLUGS:
        raise HTTPException(404, "Неверный slug")
    history = get_prompt_history(db, slug, limit=50)
    return {"slug": slug, "items": [{"id": h.id, "version": h.version, "created_at": h.created_at, "preview": (h.body or "")[:200]} for h in history]}


@router.post("/prompts/{slug}/activate")
def admin_prompts_activate(slug: str, body: PromptActivate, admin: dict = Depends(_get_current_admin), db=Depends(_get_db)):
    """Сделать указанную версию промпта активной."""
    if slug not in _PROMPT_SLUGS:
        raise HTTPException(404, "Неверный slug")
    if not set_active_prompt(db, slug, body.version_id):
        raise HTTPException(400, "Версия не найдена")
    write_audit_log(db, admin.get("sub"), admin.get("email", "") or "", "prompt_activate", "prompt", body.version_id, details=slug, ip="")
    return {"ok": True}


@router.post("/prompts/{slug}/reset")
def admin_prompts_reset(slug: str, admin: dict = Depends(_get_current_admin), db=Depends(_get_db)):
    """Сбросить промпт на встроенный (деактивировать кастомные версии)."""
    if slug not in _PROMPT_SLUGS:
        raise HTTPException(404, "Неверный slug")
    reset_prompt_to_builtin(db, slug)
    write_audit_log(db, admin.get("sub"), admin.get("email", "") or "", "prompt_reset", "prompt", None, details=slug, ip="")
    return {"ok": True}


# ─── Reports ───────────────────────────────────────────────────────────────────

REPORTS_META = [
    {"id": "registrations_by_day", "name": "Регистрации по дням", "description": "Число новых пользователей по дате", "params": ["date_from", "date_to"]},
    {"id": "tier_distribution", "name": "Распределение по тарифам", "description": "Количество пользователей по tier", "params": []},
    {"id": "revenue_by_period", "name": "Выручка за период", "description": "Сумма успешных транзакций по дням", "params": ["date_from", "date_to"]},
    {"id": "masterings_by_day", "name": "Мастеринги по дням", "description": "Число завершённых задач по дате", "params": ["date_from", "date_to"]},
    {"id": "avg_lufs_by_style", "name": "Средний LUFS по стилю", "description": "До/после по стилю мастеринга", "params": []},
    {"id": "ai_usage_by_type", "name": "Использование AI по типу и тарифу", "description": "Запросы AI по типу (recommend, report, chat и т.д.)", "params": ["date_from", "date_to"]},
    {"id": "errors_failures", "name": "Ошибки мастеринга", "description": "Задачи со статусом error", "params": ["date_from", "date_to"]},
    {"id": "popular_styles", "name": "Популярные стили", "description": "Топ стилей по количеству использований", "params": []},
    {"id": "user_activity", "name": "Активность пользователей", "description": "Уникальные пользователи по дням (мастеринг/AI)", "params": ["date_from", "date_to"]},
    {"id": "export_raw", "name": "Экспорт для аналитики", "description": "Сырые данные для внешнего BI (CSV)", "params": ["date_from", "date_to"]},
]


def _parse_dates(date_from: Optional[str], date_to: Optional[str]):
    """Вернуть (ts_from, ts_to) в секундах или (None, None)."""
    def parse(s):
        if not s:
            return None
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return time.mktime(dt.timetuple())
        except Exception:
            return None
    return parse(date_from), parse(date_to)


def _run_report(db, report_id: str, date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    """Выполнить отчёт по id. Возвращает dict с полями отчёта."""
    ts_from, ts_to = _parse_dates(date_from, date_to)
    if report_id == "registrations_by_day":
        q = db.query(User).filter(User.created_at != None)
        if ts_from:
            q = q.filter(User.created_at >= ts_from)
        if ts_to:
            q = q.filter(User.created_at <= ts_to + 86400)
        users = q.all()
        by_day: dict = {}
        for u in users:
            day = datetime.fromtimestamp(u.created_at).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + 1
        rows = [{"date": k, "count": v} for k, v in sorted(by_day.items())]
        return {"rows": rows, "total": len(users)}
    if report_id == "tier_distribution":
        q = db.query(User.tier).distinct()
        tiers = [r[0] for r in q.all()]
        out = []
        for t in tiers or ["free", "pro", "studio"]:
            cnt = db.query(User).filter(User.tier == t).count()
            out.append({"tier": t, "count": cnt})
        return {"rows": out}
    if report_id == "revenue_by_period":
        q = db.query(Transaction).filter(Transaction.status == "succeeded")
        if ts_from:
            q = q.filter(Transaction.created_at >= ts_from)
        if ts_to:
            q = q.filter(Transaction.created_at <= ts_to + 86400)
        txs = q.all()
        by_day: dict = {}
        for t in txs:
            day = datetime.fromtimestamp(t.created_at).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + t.amount
        rows = [{"date": k, "amount": round(v, 2)} for k, v in sorted(by_day.items())]
        return {"rows": rows, "total_amount": round(sum(t.amount for t in txs), 2)}
    if report_id == "masterings_by_day":
        by_day: dict = {}
        if DB_AVAILABLE and MasteringJobEvent is not None:
            q = db.query(MasteringJobEvent).filter(MasteringJobEvent.status == "done", MasteringJobEvent.completed_at != None)
            if ts_from:
                q = q.filter(MasteringJobEvent.completed_at >= ts_from)
            if ts_to:
                q = q.filter(MasteringJobEvent.completed_at <= ts_to + 86400)
            for r in q.all():
                day = datetime.fromtimestamp(r.completed_at).strftime("%Y-%m-%d")
                by_day[day] = by_day.get(day, 0) + 1
        q = db.query(MasteringRecord).filter(MasteringRecord.created_at != None)
        if ts_from:
            q = q.filter(MasteringRecord.created_at >= ts_from)
        if ts_to:
            q = q.filter(MasteringRecord.created_at <= ts_to + 86400)
        for r in q.all():
            day = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + 1
        rows = [{"date": k, "count": v} for k, v in sorted(by_day.items())]
        return {"rows": rows}
    if report_id == "avg_lufs_by_style":
        q = db.query(MasteringRecord).filter(MasteringRecord.before_lufs != None, MasteringRecord.after_lufs != None)
        recs = q.all()
        by_style: dict = {}
        for r in recs:
            s = r.style or "standard"
            if s not in by_style:
                by_style[s] = {"before": [], "after": [], "target": []}
            by_style[s]["before"].append(r.before_lufs)
            by_style[s]["after"].append(r.after_lufs)
            if r.target_lufs is not None:
                by_style[s]["target"].append(r.target_lufs)
        rows = []
        for style, data in by_style.items():
            rows.append({
                "style": style,
                "avg_before_lufs": round(sum(data["before"]) / len(data["before"]), 2) if data["before"] else None,
                "avg_after_lufs": round(sum(data["after"]) / len(data["after"]), 2) if data["after"] else None,
                "count": len(data["before"]),
            })
        return {"rows": sorted(rows, key=lambda x: -x["count"])}
    if report_id == "ai_usage_by_type":
        if not DB_AVAILABLE or AiUsageLog is None:
            return {"rows": [], "message": "Таблица ai_usage_log недоступна"}
        q = db.query(AiUsageLog).filter(AiUsageLog.type != None)
        if ts_from:
            q = q.filter(AiUsageLog.created_at >= ts_from)
        if ts_to:
            q = q.filter(AiUsageLog.created_at <= ts_to + 86400)
        logs = q.all()
        by_type_tier: dict = {}
        for L in logs:
            key = (L.type or "unknown", L.tier or "free")
            by_type_tier[key] = by_type_tier.get(key, 0) + 1
        rows = [{"type": k[0], "tier": k[1], "count": v} for k, v in sorted(by_type_tier.items(), key=lambda x: -x[1])]
        return {"rows": rows, "total": len(logs)}
    if report_id == "errors_failures":
        if not DB_AVAILABLE or MasteringJobEvent is None:
            return {"rows": [], "message": "Таблица mastering_job_events недоступна"}
        q = db.query(MasteringJobEvent).filter(MasteringJobEvent.status == "error")
        if ts_from:
            q = q.filter(MasteringJobEvent.completed_at >= ts_from)
        if ts_to:
            q = q.filter(MasteringJobEvent.completed_at <= ts_to + 86400)
        recs = q.all()
        rows = [{"job_id": r.job_id, "completed_at": r.completed_at, "style": r.style} for r in recs[:500]]
        return {"rows": rows, "total": len(recs)}
    if report_id == "popular_styles":
        q = db.query(MasteringRecord.style).filter(MasteringRecord.style != None)
        recs = q.all()
        cnt: dict = {}
        for r in recs:
            s = r[0] or "standard"
            cnt[s] = cnt.get(s, 0) + 1
        rows = [{"style": k, "count": v} for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
        return {"rows": rows}
    if report_id == "user_activity":
        by_day: dict = {}
        if DB_AVAILABLE and AiUsageLog is not None:
            q = db.query(AiUsageLog.created_at, AiUsageLog.user_id).filter(AiUsageLog.created_at != None)
            if ts_from:
                q = q.filter(AiUsageLog.created_at >= ts_from)
            if ts_to:
                q = q.filter(AiUsageLog.created_at <= ts_to + 86400)
            for r in q.all():
                if ts_from and r.created_at < ts_from or ts_to and r.created_at > ts_to + 86400:
                    continue
                day = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d")
                if day not in by_day:
                    by_day[day] = set()
                by_day[day].add(r.user_id)
        q = db.query(MasteringRecord.created_at, MasteringRecord.user_id).filter(MasteringRecord.created_at != None)
        if ts_from:
            q = q.filter(MasteringRecord.created_at >= ts_from)
        if ts_to:
            q = q.filter(MasteringRecord.created_at <= ts_to + 86400)
        for r in q.all():
            day = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d")
            if day not in by_day:
                by_day[day] = set()
            by_day[day].add(r.user_id)
        rows = [{"date": k, "unique_users": len(v)} for k, v in sorted(by_day.items())]
        return {"rows": rows}
    if report_id == "export_raw":
        return {"message": "Используйте GET /api/admin/reports/export_raw.csv?date_from=&date_to= для выгрузки CSV"}
    return {"error": "Неизвестный отчёт"}


@router.get("/reports/list")
def admin_reports_list(admin: dict = Depends(_get_current_admin)):
    """Метаданные отчётов."""
    return {"reports": REPORTS_META}


@router.get("/reports/{report_id}")
def admin_report_run(
    report_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Выполнить отчёт. Параметры: date_from, date_to (YYYY-MM-DD)."""
    if not any(r["id"] == report_id for r in REPORTS_META):
        raise HTTPException(404, "Отчёт не найден")
    return _run_report(db, report_id, date_from, date_to)


@router.post("/reports/{report_id}/summarize")
def admin_report_summarize(
    report_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Краткое резюме отчёта через LLM."""
    if not any(r["id"] == report_id for r in REPORTS_META):
        raise HTTPException(404, "Отчёт не найден")
    data = _run_report(db, report_id, date_from, date_to)
    try:
        from . import ai as ai_module
        client, _ = ai_module._get_llm_client()
        if not client:
            return {"summary": "LLM не настроен. Настройте API-ключ в разделе LLM.", "data": data}
        import json as _json
        text = _json.dumps(data, ensure_ascii=False, indent=0)[:3000]
        reply = ai_module.chat_assistant(
            [{"role": "user", "content": f"По данным отчёта дай 2-3 коротких предложения на русском (выводы, тренды):\n{text}"}],
            None,
        )
        return {"summary": reply, "data": data}
    except Exception as e:
        return {"summary": f"Ошибка: {e}", "data": data}


@router.get("/reports/export_raw.csv")
def admin_report_export_csv(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Экспорт сырых данных (users, transactions, mastering_records, ai_usage) в CSV по периоду."""
    ts_from, ts_to = _parse_dates(date_from, date_to)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["entity", "id", "data"])
    if ts_from or ts_to:
        q = db.query(User).filter(User.created_at != None)
        if ts_from:
            q = q.filter(User.created_at >= ts_from)
        if ts_to:
            q = q.filter(User.created_at <= ts_to + 86400)
        for u in q.limit(5000).all():
            w.writerow(["user", u.id, json.dumps({"email": u.email, "tier": u.tier, "created_at": u.created_at}, ensure_ascii=False)])
    else:
        for u in db.query(User).limit(2000).all():
            w.writerow(["user", u.id, json.dumps({"email": u.email, "tier": u.tier, "created_at": u.created_at}, ensure_ascii=False)])
    q = db.query(Transaction).filter(Transaction.status == "succeeded")
    if ts_from:
        q = q.filter(Transaction.created_at >= ts_from)
    if ts_to:
        q = q.filter(Transaction.created_at <= ts_to + 86400)
    for t in q.limit(5000).all():
        w.writerow(["transaction", t.id, json.dumps({"user_id": t.user_id, "amount": t.amount, "tier": t.tier, "created_at": t.created_at}, ensure_ascii=False)])
    q = db.query(MasteringRecord).filter(MasteringRecord.created_at != None)
    if ts_from:
        q = q.filter(MasteringRecord.created_at >= ts_from)
    if ts_to:
        q = q.filter(MasteringRecord.created_at <= ts_to + 86400)
    for r in q.limit(5000).all():
        w.writerow(["mastering_record", r.id, json.dumps({"user_id": r.user_id, "style": r.style, "before_lufs": r.before_lufs, "after_lufs": r.after_lufs, "created_at": r.created_at}, ensure_ascii=False)])
    if DB_AVAILABLE and AiUsageLog is not None:
        q = db.query(AiUsageLog)
        if ts_from:
            q = q.filter(AiUsageLog.created_at >= ts_from)
        if ts_to:
            q = q.filter(AiUsageLog.created_at <= ts_to + 86400)
        for L in q.limit(5000).all():
            w.writerow(["ai_usage", L.id, json.dumps({"type": L.type, "tier": L.tier, "created_at": L.created_at}, ensure_ascii=False)])
    return StreamingResponse(
        io.BytesIO(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=export_raw.csv"},
    )


def _admin_llm_settings(_mask):
    """Настройки LLM для админки: подключение и промпты (эффективные из БД/config)."""
    connected = False
    model_display = "—"
    try:
        from . import ai as ai_module
        client, model = ai_module._get_llm_client()
        connected = bool(client and model)
        model_display = model if (client and model) else "—"
        prompts = ai_module.get_effective_prompts()
    except Exception:
        prompts = {"recommend": "", "report": "", "nl_config": "", "chat": ""}

    def _s(k, d=""):
        v = _effective_setting(k)
        return (v or d) if v is not None else d

    return {
        "backend": _s("ai_backend", "openai") or "openai",
        "connected": connected,
        "model_display": model_display,
        "openai_key_set": bool((_s("openai_api_key") or "").strip()),
        "openai_key_masked": _mask(_s("openai_api_key") or ""),
        "anthropic_key_set": bool((_s("anthropic_api_key") or "").strip()),
        "deepseek_key_set": bool((_s("deepseek_api_key") or "").strip()),
        "deepseek_key_masked": _mask(_s("deepseek_api_key") or ""),
        "deepseek_base_url": (_s("deepseek_base_url") or "https://api.deepseek.com").strip() or "https://api.deepseek.com",
        "deepseek_model": (_s("deepseek_model") or "deepseek-chat").strip() or "deepseek-chat",
        "ai_limit_free": _effective_setting("ai_limit_free", 5),
        "ai_limit_pro": _effective_setting("ai_limit_pro", 50),
        "ai_limit_studio": _effective_setting("ai_limit_studio", -1),
        "llm_guard_enabled": _effective_setting("llm_guard_enabled", True),
        "llm_guard_max_length_chat": _effective_setting("llm_guard_max_length_chat", 2000),
        "llm_guard_max_length_nl": _effective_setting("llm_guard_max_length_nl", 500),
        "llm_guard_forbidden_substrings": _effective_setting("llm_guard_forbidden_substrings") or [],
        "llm_guard_forbidden_regex": (_effective_setting("llm_guard_forbidden_regex") or "").strip(),
        "llm_guard_truncate_on_long": _effective_setting("llm_guard_truncate_on_long", True),
        "llm_guard_block_role_system": _effective_setting("llm_guard_block_role_system", True),
        "prompts": prompts,
        "prompt_recommend_custom": bool((_effective_setting("ai_prompt_recommend") or "").strip() if _effective_setting("ai_prompt_recommend") is not None else False),
        "prompt_report_custom": bool((_effective_setting("ai_prompt_report") or "").strip() if _effective_setting("ai_prompt_report") is not None else False),
        "prompt_nl_config_custom": bool((_effective_setting("ai_prompt_nl_config") or "").strip() if _effective_setting("ai_prompt_nl_config") is not None else False),
        "prompt_chat_custom": bool((_effective_setting("ai_prompt_chat") or "").strip() if _effective_setting("ai_prompt_chat") is not None else False),
    }


# ─── CSV-экспорт (P39) ───────────────────────────────────────────────────────

def _make_csv(headers: list, rows: list) -> bytes:
    """Генерирует CSV-файл в памяти и возвращает bytes (UTF-8 с BOM для Excel)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")  # BOM для Excel


@router.get("/users/export.csv")
def admin_users_export_csv(
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Скачать всех пользователей как CSV-файл. P39."""
    import datetime
    if not DB_AVAILABLE or db is None:
        raise HTTPException(503, "База данных недоступна")
    users = db.query(User).order_by(User.id).all()

    def _fmt_ts(ts):
        if ts is None:
            return ""
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")

    headers = ["id", "email", "tier", "is_admin", "is_blocked",
               "subscription_status", "subscription_expires_at",
               "created_at", "last_login_at"]
    rows = [
        [
            u.id, u.email, u.tier,
            "1" if getattr(u, "is_admin", False) else "0",
            "1" if getattr(u, "is_blocked", False) else "0",
            getattr(u, "subscription_status", "none") or "none",
            _fmt_ts(getattr(u, "subscription_expires_at", None)),
            _fmt_ts(u.created_at),
            _fmt_ts(getattr(u, "last_login_at", None)),
        ]
        for u in users
    ]
    content = _make_csv(headers, rows)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=users_export.csv"},
    )


@router.get("/transactions/export.csv")
def admin_transactions_export_csv(
    admin: dict = Depends(_get_current_admin),
    db=Depends(_get_db),
):
    """Скачать все транзакции как CSV-файл. P39."""
    import datetime
    if not DB_AVAILABLE or db is None:
        raise HTTPException(503, "База данных недоступна")
    txs = db.query(Transaction).order_by(Transaction.id).all()

    def _fmt_ts(ts):
        if ts is None:
            return ""
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")

    headers = ["id", "user_id", "amount", "currency", "tier", "payment_system",
               "status", "external_id", "description", "created_at", "updated_at"]
    rows = [
        [
            t.id, t.user_id,
            str(getattr(t, "amount", "") or ""),
            getattr(t, "currency", "RUB") or "RUB",
            getattr(t, "tier", "") or "",
            getattr(t, "payment_system", "") or "",
            getattr(t, "status", "") or "",
            getattr(t, "external_id", "") or "",
            getattr(t, "description", "") or "",
            _fmt_ts(t.created_at),
            _fmt_ts(getattr(t, "updated_at", None)),
        ]
        for t in txs
    ]
    content = _make_csv(headers, rows)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=transactions_export.csv"},
    )


# ─── DB Backup (P50) ─────────────────────────────────────────────────────────

@router.get("/backup/db")
def admin_backup_db(admin: dict = Depends(_get_current_admin)):
    """Скачать резервную копию SQLite базы данных. P50.

    Использует SQLite VACUUM INTO для горячего бэкапа без блокировок.
    """
    import sqlite3 as _sqlite3
    import datetime as _dt
    from .database import DATABASE_URL

    if not DATABASE_URL or not DATABASE_URL.startswith("sqlite"):
        raise HTTPException(400, "Backup доступен только для SQLite")

    db_path = DATABASE_URL.replace("sqlite:///", "").replace("sqlite://", "")
    if not os.path.isabs(db_path):
        raise HTTPException(400, "Не удалось определить путь к базе данных")

    if not os.path.exists(db_path):
        raise HTTPException(404, "Файл базы данных не найден")

    import tempfile as _tmp
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_name = f"magic_master_backup_{stamp}.sqlite3"

    try:
        # VACUUM INTO — «горячая» копия без блокировки записи
        tmp = _tmp.NamedTemporaryFile(delete=False, suffix=".sqlite3")
        tmp_path = tmp.name
        tmp.close()
        src_conn = _sqlite3.connect(db_path, check_same_thread=False)
        src_conn.execute(f"VACUUM INTO '{tmp_path}'")
        src_conn.close()
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Ошибка создания бэкапа: {e}")

    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={backup_name}"},
    )
