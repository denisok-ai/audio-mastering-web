"""YooKassa payment integration (P23).

Эндпоинты:
  POST /api/payments/create   — создать платёж, вернуть ссылку на оплату
  POST /api/payments/webhook  — получить уведомление от YooKassa (payment.succeeded)

Конфигурация:
  MAGIC_MASTER_YOOKASSA_SHOP_ID     — ИД магазина
  MAGIC_MASTER_YOOKASSA_SECRET_KEY  — секретный ключ
  MAGIC_MASTER_YOOKASSA_RETURN_URL  — URL редиректа после оплаты

Тарифные планы и цены в TIER_PRICES (редактируйте под свои нужды).
"""
import json
import logging
import time
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from .auth import decode_token, extract_bearer_token
from .config import settings
from .database import (
    DB_AVAILABLE,
    SessionLocal,
    Transaction,
    User,
    create_transaction,
)

logger = logging.getLogger("payments")

router = APIRouter(prefix="/api/payments", tags=["payments"])

# ─── Цены по тарифам (в рублях) ───────────────────────────────────────────────
TIER_PRICES: dict[str, dict] = {
    "pro_month": {
        "tier": "pro",
        "label": "Pro — 1 месяц",
        "amount": "990.00",
        "currency": "RUB",
        "period_days": 30,
    },
    "pro_year": {
        "tier": "pro",
        "label": "Pro — 1 год",
        "amount": "7900.00",
        "currency": "RUB",
        "period_days": 365,
    },
    "studio_month": {
        "tier": "studio",
        "label": "Studio — 1 месяц",
        "amount": "2990.00",
        "currency": "RUB",
        "period_days": 30,
    },
    "studio_year": {
        "tier": "studio",
        "label": "Studio — 1 год",
        "amount": "24900.00",
        "currency": "RUB",
        "period_days": 365,
    },
}


def _is_configured() -> bool:
    return bool(
        getattr(settings, "yookassa_shop_id", "").strip()
        and getattr(settings, "yookassa_secret_key", "").strip()
    )


def _get_yookassa():
    """Инициализировать YooKassa SDK. Возвращает модуль payment или None."""
    if not _is_configured():
        return None
    try:
        from yookassa import Configuration, Payment
        Configuration.account_id = settings.yookassa_shop_id.strip()
        Configuration.secret_key = settings.yookassa_secret_key.strip()
        return Payment
    except ImportError:
        logger.warning("Пакет yookassa не установлен. Запустите: pip install yookassa")
        return None


def _get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    token = extract_bearer_token(authorization)
    if not token:
        return None
    return decode_token(token)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class CreatePaymentRequest(BaseModel):
    plan: str               # ключ из TIER_PRICES: "pro_month", "studio_year" и т.д.
    return_url: Optional[str] = None   # переопределить URL редиректа


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@router.get("/plans")
def get_plans():
    """Список доступных тарифных планов с ценами (публичный)."""
    return {"plans": TIER_PRICES}


@router.post("/create")
async def create_payment(
    body: CreatePaymentRequest,
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Создать платёж в YooKassa. Возвращает confirmation_url для редиректа.

    Требуется авторизация (JWT).
    """
    if not user:
        raise HTTPException(401, "Требуется авторизация для оплаты")

    plan = TIER_PRICES.get(body.plan)
    if not plan:
        raise HTTPException(400, f"Неизвестный план: {body.plan}. Доступны: {list(TIER_PRICES.keys())}")

    Payment = _get_yookassa()
    return_url = (body.return_url or getattr(settings, "yookassa_return_url", "")
                  or "http://localhost:8000/pricing").strip()

    user_id = int(user.get("sub", 0))
    idempotency_key = str(_uuid.uuid4())

    if Payment is None:
        # YooKassa не настроен — возвращаем заглушку для разработки
        logger.warning("YooKassa не настроен, возвращается тестовая ссылка")
        if DB_AVAILABLE and SessionLocal:
            db = SessionLocal()
            try:
                create_transaction(
                    db,
                    user_id=user_id,
                    amount=float(plan["amount"]),
                    tier=plan["tier"],
                    payment_system="yookassa",
                    currency=plan["currency"],
                    external_id="demo-" + idempotency_key[:8],
                    status="pending",
                    description=f"[DEMO] {plan['label']}",
                )
            finally:
                db.close()
        return {
            "status": "demo",
            "confirmation_url": return_url,
            "plan": plan,
            "note": "YooKassa не настроен. Задайте MAGIC_MASTER_YOOKASSA_SHOP_ID и MAGIC_MASTER_YOOKASSA_SECRET_KEY.",
        }

    try:
        payment = Payment.create(
            {
                "amount": {"value": plan["amount"], "currency": plan["currency"]},
                "confirmation": {"type": "redirect", "return_url": return_url},
                "capture": True,
                "description": f"{plan['label']} для {user.get('email', 'пользователя')}",
                "metadata": {
                    "user_id": user_id,
                    "plan": body.plan,
                    "tier": plan["tier"],
                    "period_days": plan["period_days"],
                },
            },
            idempotency_key,
        )
    except Exception as exc:
        logger.error("YooKassa create_payment error: %s", exc)
        raise HTTPException(502, f"Ошибка создания платежа: {exc}")

    # Сохранить транзакцию со статусом pending
    if DB_AVAILABLE and SessionLocal:
        db = SessionLocal()
        try:
            create_transaction(
                db,
                user_id=user_id,
                amount=float(plan["amount"]),
                tier=plan["tier"],
                payment_system="yookassa",
                currency=plan["currency"],
                external_id=payment.id,
                status="pending",
                description=plan["label"],
            )
        finally:
            db.close()

    return {
        "payment_id": payment.id,
        "status": payment.status,
        "confirmation_url": payment.confirmation.confirmation_url,
        "plan": plan,
    }


def _webhook_client_ip(request: Request) -> str:
    """IP клиента для webhook (учёт X-Forwarded-For за прокси)."""
    forwarded = (request.headers.get("x-forwarded-for") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return getattr(request.client, "host", "") or ""
    return ""


@router.post("/webhook")
async def yookassa_webhook(request: Request):
    """Webhook от YooKassa. Вызывается при изменении статуса платежа.

    YooKassa POST-ит JSON с объектом уведомления (type: notification, event: payment.succeeded).
    P56: при заданном MAGIC_MASTER_YOOKASSA_WEBHOOK_IP_WHITELIST (IP через запятую)
    принимаем запросы только с этих адресов; иначе проверка отключена.
    """
    # P56: опциональная проверка IP whitelist
    whitelist_str = getattr(settings, "yookassa_webhook_ip_whitelist", "") or ""
    if whitelist_str.strip():
        allowed_ips = {ip.strip() for ip in whitelist_str.split(",") if ip.strip()}
        client_ip = _webhook_client_ip(request)
        if client_ip not in allowed_ips:
            logger.warning("YooKassa webhook: отклонён запрос с IP %s (не в whitelist)", client_ip)
            raise HTTPException(403, "Доступ запрещён")

    try:
        body = await request.json()
    except Exception as e:
        logger.error("YooKassa webhook: некорректный JSON error=%s", str(e)[:200])
        raise HTTPException(400, "Некорректный JSON")

    event = body.get("event")
    obj = body.get("object", {})

    logger.info("YooKassa webhook: event=%s id=%s", event, obj.get("id"))

    if event == "payment.succeeded":
        payment_id = obj.get("id")
        metadata = obj.get("metadata") or {}
        user_id = metadata.get("user_id")
        tier = metadata.get("tier")
        period_days = int(metadata.get("period_days", 30))
        amount_obj = obj.get("amount") or {}
        amount_val = float(amount_obj.get("value", 0))
        currency_val = (amount_obj.get("currency") or "RUB").strip()

        if not user_id or not tier:
            logger.warning("Webhook: нет user_id или tier в metadata")
            return {"status": "ignored"}

        if DB_AVAILABLE and SessionLocal:
            db = SessionLocal()
            try:
                # Обновить транзакцию
                tx = db.query(Transaction).filter(Transaction.external_id == payment_id).first()
                if tx:
                    tx.status = "succeeded"
                    db.commit()
                else:
                    create_transaction(
                        db,
                        user_id=int(user_id),
                        amount=amount_val,
                        tier=tier,
                        payment_system="yookassa",
                        external_id=payment_id,
                        status="succeeded",
                        description=metadata.get("plan", "YooKassa"),
                    )

                # Апгрейд пользователя
                u = db.query(User).filter(User.id == int(user_id)).first()
                if u:
                    u.tier = tier
                    expires = time.time() + period_days * 86400
                    u.subscription_expires_at = expires
                    u.subscription_status = "active"
                    db.commit()
                    logger.info("Пользователь %s апгрейжен до %s (до %s)", user_id, tier, expires)

                    # Письмо об активации (P22)
                    try:
                        from .mailer import send_subscription_activated_email
                        send_subscription_activated_email(
                            u.email, tier, u.subscription_expires_at
                        )
                    except Exception:  # noqa: BLE001
                        pass

                    # P51: Telegram-уведомление об успешной оплате
                    try:
                        from .notifier import notify_payment
                        notify_payment(u.email, amount_val, currency_val, tier)
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                db.close()

    elif event == "payment.canceled":
        payment_id = obj.get("id")
        if DB_AVAILABLE and SessionLocal and payment_id:
            db = SessionLocal()
            try:
                tx = db.query(Transaction).filter(Transaction.external_id == payment_id).first()
                if tx:
                    tx.status = "failed"
                    db.commit()
            finally:
                db.close()

    return {"status": "ok"}
