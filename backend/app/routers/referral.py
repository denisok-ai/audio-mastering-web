# @file routers/referral.py
# @description Реферальная программа: ссылка и статистика.

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..database import (
    DB_AVAILABLE,
    Referral,
    User,
    ensure_user_referral_code,
    get_db,
)
from ..deps import get_current_user_optional

router = APIRouter(prefix="/api/referral", tags=["referral"])


def _require_user(user: Optional[dict]) -> dict:
    if not user or not user.get("sub"):
        raise HTTPException(401, "Требуется вход в аккаунт")
    return user


def _uid(user: dict) -> int:
    return int(user["sub"])


class ReferralLinkResponse(BaseModel):
    referral_code: str
    referral_url: str
    bot_referral_url: str


class ReferralRow(BaseModel):
    invitee_id: int
    status: str
    created_at: float
    rewarded_at: Optional[float] = None


class ReferralStatsResponse(BaseModel):
    referrals: list[ReferralRow]
    total_invited: int
    total_rewarded: int


@router.get("/my-link", response_model=ReferralLinkResponse)
def api_referral_my_link(
    user: Optional[dict] = Depends(get_current_user_optional),
    db=Depends(get_db),
):
    u = _require_user(user)
    if not DB_AVAILABLE or db is None or User is None:
        raise HTTPException(503, "База данных недоступна")
    row = db.query(User).filter(User.id == _uid(u)).first()
    if not row:
        raise HTTPException(404, "Пользователь не найден")
    ensure_user_referral_code(db, row)
    db.refresh(row)
    code = row.referral_code or ""
    base = getattr(settings, "public_base_url", "") or "https://magicmaster.pro"
    base = base.rstrip("/")
    bot = getattr(settings, "user_bot_telegram_url", "") or "https://t.me/magicmasterpro_user_bot"
    bot = bot.rstrip("/")
    return ReferralLinkResponse(
        referral_code=code,
        referral_url=f"{base}/register?ref={code}",
        bot_referral_url=f"{bot}?start=ref_{code}",
    )


@router.get("/stats", response_model=ReferralStatsResponse)
def api_referral_stats(
    user: Optional[dict] = Depends(get_current_user_optional),
    db=Depends(get_db),
):
    u = _require_user(user)
    if not DB_AVAILABLE or db is None or Referral is None:
        raise HTTPException(503, "База данных недоступна")
    uid = _uid(u)
    rows = db.query(Referral).filter(Referral.inviter_id == uid).order_by(Referral.created_at.desc()).all()
    referrals = [
        ReferralRow(
            invitee_id=int(r.invitee_id),
            status=str(r.status),
            created_at=float(r.created_at),
            rewarded_at=float(r.rewarded_at) if r.rewarded_at else None,
        )
        for r in rows
    ]
    rewarded = sum(1 for r in rows if r.status == "rewarded")
    return ReferralStatsResponse(
        referrals=referrals,
        total_invited=len(rows),
        total_rewarded=rewarded,
    )
