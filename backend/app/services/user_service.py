"""Сервис управления пользователями (admin).

Содержит чистую бизнес-логику: CRUD пользователей, управление подписками,
массовые операции. Роутер (admin.py) вызывает эти функции.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import HTTPException

from ..database import (
    MasteringRecord,
    SavedPreset,
    Transaction,
    User,
    create_transaction,
    write_audit_log,
)


# ─── Serializers ──────────────────────────────────────────────────────────────

def user_to_dict(u: "User") -> dict:
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


def tx_to_dict(t: "Transaction") -> dict:
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


# ─── Queries ──────────────────────────────────────────────────────────────────

def list_users(
    db,
    search: str = "",
    tier: str = "",
    blocked: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
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
        "users": [user_to_dict(u) for u in users],
    }


def get_user_detail(db, user_id: int) -> dict:
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
        **user_to_dict(u),
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
        "recent_transactions": [tx_to_dict(t) for t in txs],
    }


def patch_user(
    db,
    user_id: int,
    tier: Optional[str],
    is_blocked: Optional[bool],
    is_admin: Optional[bool],
    admin_payload: dict,
    client_ip: str = "",
) -> dict:
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    changes: list[str] = []
    if tier is not None:
        if tier not in ("free", "pro", "studio"):
            raise HTTPException(400, "tier должен быть free, pro или studio")
        changes.append(f"tier: {u.tier} → {tier}")
        u.tier = tier
    if is_blocked is not None:
        changes.append(f"blocked: {u.is_blocked} → {is_blocked}")
        u.is_blocked = is_blocked
    if is_admin is not None:
        changes.append(f"is_admin: {u.is_admin} → {is_admin}")
        u.is_admin = is_admin
    db.commit()
    db.refresh(u)
    write_audit_log(
        db,
        int(admin_payload.get("sub", 0) or 0),
        admin_payload.get("email", ""),
        action="patch_user",
        target_type="user",
        target_id=user_id,
        details="; ".join(changes) or "no changes",
        ip=client_ip,
    )
    return user_to_dict(u)


def delete_user(db, user_id: int, admin_payload: dict, client_ip: str = "") -> dict:
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    if str(user_id) == str(admin_payload.get("sub")):
        raise HTTPException(400, "Нельзя удалить собственную учётную запись через API")
    user_email = u.email
    db.delete(u)
    db.commit()
    write_audit_log(
        db,
        int(admin_payload.get("sub", 0) or 0),
        admin_payload.get("email", ""),
        action="delete_user",
        target_type="user",
        target_id=user_id,
        details=f"email: {user_email}",
        ip=client_ip,
    )
    return {"deleted": True, "user_id": user_id}


def bulk_action(
    db,
    user_ids: list[int],
    action: str,
    tier: Optional[str],
    admin_payload: dict,
    client_ip: str = "",
) -> dict:
    if not user_ids:
        raise HTTPException(400, "user_ids не может быть пустым")
    if action not in ("block", "unblock", "delete", "set_tier"):
        raise HTTPException(400, "action: block | unblock | delete | set_tier")
    if action == "set_tier" and tier not in ("free", "pro", "studio"):
        raise HTTPException(400, "tier должен быть free, pro или studio")

    admin_id = str(admin_payload.get("sub"))
    affected = 0
    skipped = 0
    errors: list[Any] = []

    for uid in user_ids:
        if action == "delete" and str(uid) == admin_id:
            skipped += 1
            continue
        u = db.query(User).filter(User.id == uid).first()
        if not u:
            skipped += 1
            continue
        try:
            if action == "block":
                u.is_blocked = True
            elif action == "unblock":
                u.is_blocked = False
            elif action == "set_tier":
                u.tier = tier
            elif action == "delete":
                db.delete(u)
            affected += 1
        except Exception as e:
            errors.append({"user_id": uid, "error": str(e)[:100]})
            skipped += 1

    db.commit()
    write_audit_log(
        db,
        int(admin_payload.get("sub", 0) or 0),
        admin_payload.get("email", ""),
        action=f"bulk_{action}",
        target_type="user",
        details=f"ids={user_ids[:20]}; affected={affected}; tier={tier or ''}",
        ip=client_ip,
    )
    return {"action": action, "affected": affected, "skipped": skipped, "errors": errors}


def set_subscription(db, user_id: int, tier: str, expires_at: Optional[float], amount: float, description: str) -> dict:
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "Пользователь не найден")
    if tier not in ("free", "pro", "studio"):
        raise HTTPException(400, "tier должен быть free, pro или studio")
    u.tier = tier
    u.subscription_expires_at = expires_at
    u.subscription_status = "active" if tier != "free" else "none"
    db.commit()
    tx = create_transaction(
        db,
        user_id=user_id,
        amount=amount,
        tier=tier,
        payment_system="manual",
        status="succeeded",
        description=description or f"Ручное назначение тарифа {tier}",
    )
    db.refresh(u)
    try:
        from ..mailer import send_subscription_activated_email
        import asyncio as _asyncio
        _asyncio.get_event_loop().run_in_executor(
            None, send_subscription_activated_email, u.email, tier, expires_at
        )
    except Exception:  # noqa: BLE001
        pass
    return {"user": user_to_dict(u), "transaction": tx_to_dict(tx) if tx else None}
