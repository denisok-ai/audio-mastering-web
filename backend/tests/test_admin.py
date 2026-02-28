"""Тесты для Admin API и Payments API (P26).

Запуск:
    cd backend && python3 -m pytest tests/test_admin.py -v
"""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from httpx import AsyncClient, ASGITransport
from app.main import app, _rate_limits


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_limits():
    _rate_limits.clear()
    yield
    _rate_limits.clear()


async def _make_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _register_and_login(ac: AsyncClient, email: str, password: str = "test1234") -> str:
    """Зарегистрировать и получить токен."""
    r = await ac.post("/api/auth/register", json={"email": email, "password": password})
    if r.status_code == 400:
        # уже существует — логинимся
        r = await ac.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, f"Login/register failed: {r.text}"
    return r.json()["access_token"]


async def _make_admin(ac: AsyncClient, email: str, password: str = "test1234") -> str:
    """Зарегистрировать, сделать admin через БД, получить токен."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    token = await _register_and_login(ac, email, password)
    if DB_AVAILABLE and SessionLocal:
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email.lower()).first()
            if u:
                u.is_admin = True
                db.commit()
        finally:
            db.close()
    # Перелогиниться чтобы получить токен с is_admin=True
    r = await ac.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    return r.json()["access_token"]


# ─── Public API ───────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_public_news_empty():
    """GET /api/news должен отдавать пустой список при отсутствии новостей."""
    async with await _make_client() as ac:
        r = await ac.get("/api/news")
    assert r.status_code == 200
    data = r.json()
    assert "posts" in data
    assert isinstance(data["posts"], list)


@pytest.mark.anyio
async def test_payments_plans():
    """GET /api/payments/plans — публичный список тарифов."""
    async with await _make_client() as ac:
        r = await ac.get("/api/payments/plans")
    assert r.status_code == 200
    data = r.json()
    assert "plans" in data
    assert "pro_month" in data["plans"]
    assert "studio_month" in data["plans"]
    plan = data["plans"]["pro_month"]
    assert "amount" in plan
    assert "tier" in plan


# ─── Admin auth guard ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_stats_requires_auth():
    """GET /api/admin/stats без токена → 401."""
    async with await _make_client() as ac:
        r = await ac.get("/api/admin/stats")
    assert r.status_code == 401


@pytest.mark.anyio
async def test_admin_stats_requires_admin_role():
    """GET /api/admin/stats с обычным токеном → 403."""
    async with await _make_client() as ac:
        token = await _register_and_login(ac, "regular@test.example")
        r = await ac.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403


# ─── Admin stats ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_stats_ok():
    """GET /api/admin/stats с admin токеном → 200 с ожидаемыми полями."""
    from app.database import DB_AVAILABLE
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_stats@test.example")
        r = await ac.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert "users" in data
    assert "revenue" in data
    assert "news" in data
    assert "total" in data["users"]
    assert "by_tier" in data["users"]


# ─── Admin user management ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_list_users():
    """GET /api/admin/users → список пользователей."""
    from app.database import DB_AVAILABLE
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_list@test.example")
        r = await ac.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert "users" in data
    assert "total" in data
    assert isinstance(data["users"], list)


@pytest.mark.anyio
async def test_admin_get_user_detail():
    """GET /api/admin/users/{id} → детали пользователя."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_detail@test.example")
        # Найти ID себя
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "admin_detail@test.example").first()
            uid = u.id
        finally:
            db.close()
        r = await ac.get(f"/api/admin/users/{uid}", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert data["email"] == "admin_detail@test.example"
    assert "mastering_count" in data
    assert "recent_records" in data


@pytest.mark.anyio
async def test_admin_patch_user_tier():
    """PATCH /api/admin/users/{id} — изменить тариф."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        admin_token = await _make_admin(ac, "admin_patch@test.example")
        # Создать второго пользователя
        await _register_and_login(ac, "target_patch@test.example")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "target_patch@test.example").first()
            uid = u.id
        finally:
            db.close()
        r = await ac.patch(
            f"/api/admin/users/{uid}",
            json={"tier": "studio"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert r.status_code == 200
    assert r.json()["tier"] == "studio"


@pytest.mark.anyio
async def test_admin_patch_user_block():
    """PATCH /api/admin/users/{id} — заблокировать пользователя."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        admin_token = await _make_admin(ac, "admin_block@test.example")
        await _register_and_login(ac, "target_block@test.example")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "target_block@test.example").first()
            uid = u.id
        finally:
            db.close()
        r = await ac.patch(
            f"/api/admin/users/{uid}",
            json={"is_blocked": True},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert r.status_code == 200
    assert r.json()["is_blocked"] is True


@pytest.mark.anyio
async def test_admin_patch_invalid_tier():
    """PATCH /api/admin/users/{id} с неверным tier → 400."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        admin_token = await _make_admin(ac, "admin_badtier@test.example")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "admin_badtier@test.example").first()
            uid = u.id
        finally:
            db.close()
        r = await ac.patch(
            f"/api/admin/users/{uid}",
            json={"tier": "enterprise"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert r.status_code == 400


# ─── Admin subscription ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_set_subscription():
    """POST /api/admin/users/{id}/subscription — назначить подписку вручную."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        admin_token = await _make_admin(ac, "admin_subs@test.example")
        await _register_and_login(ac, "target_subs@test.example")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "target_subs@test.example").first()
            uid = u.id
        finally:
            db.close()
        r = await ac.post(
            f"/api/admin/users/{uid}/subscription",
            json={"tier": "pro", "amount": 990.0, "description": "Тестовая подписка"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["user"]["tier"] == "pro"
    assert data["transaction"] is not None
    assert data["transaction"]["amount"] == 990.0


# ─── Admin transactions ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_list_transactions():
    """GET /api/admin/transactions → список транзакций."""
    from app.database import DB_AVAILABLE
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_txlist@test.example")
        r = await ac.get("/api/admin/transactions", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert "transactions" in data


@pytest.mark.anyio
async def test_admin_create_transaction():
    """POST /api/admin/transactions — добавить ручную транзакцию."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        admin_token = await _make_admin(ac, "admin_txcreate@test.example")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "admin_txcreate@test.example").first()
            uid = u.id
        finally:
            db.close()
        r = await ac.post(
            "/api/admin/transactions",
            json={"user_id": uid, "amount": 500.0, "tier": "pro", "description": "Тест"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["amount"] == 500.0
    assert data["tier"] == "pro"
    assert data["payment_system"] == "manual"


# ─── Admin news ───────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_news_crud():
    """CRUD новостей: создать → список → опубликовать → удалить."""
    from app.database import DB_AVAILABLE
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_news@test.example")
        headers = {"Authorization": f"Bearer {token}"}

        # Создать черновик
        r = await ac.post("/api/admin/news",
                          json={"title": "Тест новости", "body": "Текст", "is_published": False},
                          headers=headers)
        assert r.status_code == 200
        post_id = r.json()["id"]
        assert r.json()["is_published"] is False

        # Список (включая черновики)
        r = await ac.get("/api/admin/news", headers=headers)
        assert r.status_code == 200
        assert any(p["id"] == post_id for p in r.json()["posts"])

        # Опубликовать
        r = await ac.put(f"/api/admin/news/{post_id}",
                         json={"is_published": True}, headers=headers)
        assert r.status_code == 200
        assert r.json()["is_published"] is True

        # Публичный API должен видеть пост
        r = await ac.get("/api/news")
        assert r.status_code == 200
        assert any(p["id"] == post_id for p in r.json()["posts"])

        # Удалить
        r = await ac.delete(f"/api/admin/news/{post_id}", headers=headers)
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        # После удаления — публичный API не видит
        r = await ac.get("/api/news")
        assert not any(p["id"] == post_id for p in r.json()["posts"])


# ─── Admin campaigns ──────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_admin_campaigns_create():
    """POST /api/admin/campaigns — создать черновик кампании."""
    from app.database import DB_AVAILABLE
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_camp@test.example")
        r = await ac.post(
            "/api/admin/campaigns",
            json={"subject": "Тест", "body_html": "<p>Тест</p>", "body_text": "Тест"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["subject"] == "Тест"
    assert data["status"] == "draft"


@pytest.mark.anyio
async def test_admin_campaigns_list():
    """GET /api/admin/campaigns — список кампаний."""
    from app.database import DB_AVAILABLE
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _make_admin(ac, "admin_camplist@test.example")
        r = await ac.get("/api/admin/campaigns", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "campaigns" in r.json()


# ─── Payments ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_payment_create_requires_auth():
    """POST /api/payments/create без токена → 401."""
    async with await _make_client() as ac:
        r = await ac.post("/api/payments/create", json={"plan": "pro_month"})
    assert r.status_code == 401


@pytest.mark.anyio
async def test_payment_create_invalid_plan():
    """POST /api/payments/create с неверным plan → 400."""
    async with await _make_client() as ac:
        token = await _register_and_login(ac, "pay_badplan@test.example")
        r = await ac.post(
            "/api/payments/create",
            json={"plan": "nonexistent_plan"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_payment_create_demo_mode():
    """POST /api/payments/create без YooKassa конфига → demo ответ (не падает)."""
    from app.config import settings
    # Убеждаемся что YooKassa не настроен
    orig_shop = getattr(settings, "yookassa_shop_id", "")
    settings.yookassa_shop_id = ""
    try:
        async with await _make_client() as ac:
            token = await _register_and_login(ac, "pay_demo@test.example")
            r = await ac.post(
                "/api/payments/create",
                json={"plan": "pro_month"},
                headers={"Authorization": f"Bearer {token}"},
            )
        # Должен вернуть demo ответ, не 5xx
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert data.get("status") == "demo"
    finally:
        settings.yookassa_shop_id = orig_shop


@pytest.mark.anyio
async def test_webhook_ignored_unknown_event():
    """POST /api/payments/webhook с неизвестным event → 200 ignored."""
    async with await _make_client() as ac:
        r = await ac.post(
            "/api/payments/webhook",
            json={"event": "payment.unknown", "object": {"id": "fake-id"}},
        )
    assert r.status_code == 200


@pytest.mark.anyio
async def test_webhook_403_when_ip_not_in_whitelist():
    """P56: при заданном MAGIC_MASTER_YOOKASSA_WEBHOOK_IP_WHITELIST запрос с чужого IP → 403."""
    from app.config import settings
    orig = getattr(settings, "yookassa_webhook_ip_whitelist", "")
    settings.yookassa_webhook_ip_whitelist = "10.0.0.1,10.0.0.2"
    try:
        async with await _make_client() as ac:
            r = await ac.post(
                "/api/payments/webhook",
                json={"event": "payment.succeeded", "object": {"id": "x", "amount": {"value": "1", "currency": "RUB"}, "metadata": {"user_id": "1", "tier": "pro", "period_days": 30}}},
            )
        assert r.status_code == 403
    finally:
        settings.yookassa_webhook_ip_whitelist = orig


@pytest.mark.anyio
async def test_webhook_payment_succeeded_upgrades_user():
    """POST /api/payments/webhook с payment.succeeded → апгрейд пользователя."""
    from app.database import DB_AVAILABLE, SessionLocal, User
    if not DB_AVAILABLE:
        pytest.skip("DB недоступна")
    async with await _make_client() as ac:
        token = await _register_and_login(ac, "webhook_user@test.example")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == "webhook_user@test.example").first()
            uid = u.id
            assert u.tier == "pro"  # дефолт при регистрации
        finally:
            db.close()

        r = await ac.post(
            "/api/payments/webhook",
            json={
                "event": "payment.succeeded",
                "object": {
                    "id": "test-payment-id-123",
                    "amount": {"value": "990.00", "currency": "RUB"},
                    "metadata": {
                        "user_id": uid,
                        "tier": "studio",
                        "period_days": 30,
                        "plan": "studio_month",
                    },
                },
            },
        )
    assert r.status_code == 200

    # Проверить что тариф апгрейжен
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == uid).first()
        assert u.tier == "studio"
        assert u.subscription_expires_at is not None
        assert u.subscription_status == "active"
    finally:
        db.close()
