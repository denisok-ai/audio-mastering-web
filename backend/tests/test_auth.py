"""Тесты Auth API: профиль, смена пароля, сброс пароля, rate limit (P38).

Запуск:
    cd backend && python3 -m pytest tests/test_auth.py -v
"""
import sys
import os
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app, _rate_limits, _auth_attempts, _reset_tokens


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_limits():
    """Очищаем все rate-limit счётчики и токены между тестами."""
    _rate_limits.clear()
    _auth_attempts.clear()
    _reset_tokens.clear()
    yield
    _rate_limits.clear()
    _auth_attempts.clear()
    _reset_tokens.clear()


def _make_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _register(client, email="user@test.com", password="secret123") -> str:
    """Регистрирует пользователя, возвращает токен."""
    r = await client.post("/api/auth/register", json={"email": email, "password": password})
    assert r.status_code == 200, f"Register failed: {r.text}"
    return r.json()["access_token"]


# ─── Register / Login ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_register_success():
    async with _make_client() as ac:
        r = await ac.post("/api/auth/register", json={"email": "new@test.com", "password": "pass123"})
    assert r.status_code == 200
    d = r.json()
    assert "access_token" in d
    assert d["email"] == "new@test.com"
    assert "tier" in d


@pytest.mark.anyio
async def test_register_duplicate_email():
    async with _make_client() as ac:
        await ac.post("/api/auth/register", json={"email": "dup@test.com", "password": "pass123"})
        r = await ac.post("/api/auth/register", json={"email": "dup@test.com", "password": "other"})
    assert r.status_code == 400
    assert "уже существует" in r.json()["detail"].lower() or "exist" in r.json()["detail"].lower()


@pytest.mark.anyio
async def test_login_success():
    async with _make_client() as ac:
        await ac.post("/api/auth/register", json={"email": "login@test.com", "password": "mypass"})
        r = await ac.post("/api/auth/login", json={"email": "login@test.com", "password": "mypass"})
    assert r.status_code == 200
    assert "access_token" in r.json()


@pytest.mark.anyio
async def test_login_wrong_password():
    async with _make_client() as ac:
        await ac.post("/api/auth/register", json={"email": "wrong@test.com", "password": "correct"})
        r = await ac.post("/api/auth/login", json={"email": "wrong@test.com", "password": "wrong"})
    assert r.status_code == 401


@pytest.mark.anyio
async def test_login_unknown_email():
    async with _make_client() as ac:
        r = await ac.post("/api/auth/login", json={"email": "ghost@test.com", "password": "any"})
    assert r.status_code == 401


# ─── Auth rate limit (P33) ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_auth_rate_limit_login():
    """После 10 попыток входа с одного IP — 429."""
    from app.main import _AUTH_LIMIT_PER_MINUTE
    import time
    # Заполняем счётчик (имитируем max попыток)
    _auth_attempts["testclient"] = {"count": _AUTH_LIMIT_PER_MINUTE, "window_start": time.time()}
    async with _make_client() as ac:
        r = await ac.post("/api/auth/login", json={"email": "any@test.com", "password": "any"})
    assert r.status_code == 429
    assert "попыток" in r.json()["detail"].lower() or "429" in str(r.status_code)


@pytest.mark.anyio
async def test_auth_rate_limit_register():
    """После 10 попыток регистрации с одного IP — 429."""
    from app.main import _AUTH_LIMIT_PER_MINUTE
    import time
    _auth_attempts["testclient"] = {"count": _AUTH_LIMIT_PER_MINUTE, "window_start": time.time()}
    async with _make_client() as ac:
        r = await ac.post("/api/auth/register", json={"email": "new2@test.com", "password": "pass123"})
    assert r.status_code == 429


@pytest.mark.anyio
async def test_auth_rate_limit_resets_after_minute():
    """После истечения окна (>60с) лимит сбрасывается."""
    from app.main import _AUTH_LIMIT_PER_MINUTE
    import time
    # Окно давно истекло
    _auth_attempts["testclient"] = {"count": _AUTH_LIMIT_PER_MINUTE, "window_start": time.time() - 120}
    async with _make_client() as ac:
        r = await ac.post("/api/auth/login", json={"email": "any@test.com", "password": "any"})
    # Должен пройти rate limit (401 = неверный пароль, не 429)
    assert r.status_code == 401


# ─── /api/auth/me ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_me_requires_auth():
    async with _make_client() as ac:
        r = await ac.get("/api/auth/me")
    assert r.status_code == 401


@pytest.mark.anyio
async def test_me_returns_user_info():
    async with _make_client() as ac:
        token = await _register(ac, "me@test.com")
        r = await ac.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    d = r.json()
    assert d["email"] == "me@test.com"
    assert "tier" in d
    assert "user_id" in d


# ─── /api/auth/profile (P31) ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_profile_requires_auth():
    async with _make_client() as ac:
        r = await ac.get("/api/auth/profile")
    assert r.status_code == 401


@pytest.mark.anyio
async def test_profile_returns_full_data():
    async with _make_client() as ac:
        token = await _register(ac, "profile@test.com")
        r = await ac.get("/api/auth/profile", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    d = r.json()
    assert d["email"] == "profile@test.com"
    assert "tier" in d
    assert "subscription_status" in d
    assert "stats" in d
    assert "created_at" in d


@pytest.mark.anyio
async def test_profile_stats_initially_zero():
    async with _make_client() as ac:
        token = await _register(ac, "stats@test.com")
        r = await ac.get("/api/auth/profile", headers={"Authorization": f"Bearer {token}"})
    stats = r.json()["stats"]
    assert stats.get("total_masterings", 0) == 0


# ─── /api/auth/change-password (P34) ─────────────────────────────────────────

@pytest.mark.anyio
async def test_change_password_success():
    async with _make_client() as ac:
        token = await _register(ac, "chpass@test.com", "oldpass")
        r = await ac.post(
            "/api/auth/change-password",
            json={"old_password": "oldpass", "new_password": "newpass999"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert "изменён" in r.json()["message"].lower() or "changed" in r.json()["message"].lower()
        # Проверяем: вход со старым паролем — 401
        r2 = await ac.post("/api/auth/login", json={"email": "chpass@test.com", "password": "oldpass"})
        assert r2.status_code == 401
        # Вход с новым — 200
        r3 = await ac.post("/api/auth/login", json={"email": "chpass@test.com", "password": "newpass999"})
        assert r3.status_code == 200


@pytest.mark.anyio
async def test_change_password_wrong_old():
    async with _make_client() as ac:
        token = await _register(ac, "chpasswrong@test.com", "correct")
        r = await ac.post(
            "/api/auth/change-password",
            json={"old_password": "wrong", "new_password": "newpass"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_change_password_too_short():
    async with _make_client() as ac:
        token = await _register(ac, "chpassshort@test.com", "correct")
        r = await ac.post(
            "/api/auth/change-password",
            json={"old_password": "correct", "new_password": "ab"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert r.status_code == 422  # Validation error


@pytest.mark.anyio
async def test_change_password_requires_auth():
    async with _make_client() as ac:
        r = await ac.post(
            "/api/auth/change-password",
            json={"old_password": "a", "new_password": "b"},
        )
    assert r.status_code == 401


# ─── /api/auth/forgot-password (P35) ─────────────────────────────────────────

@pytest.mark.anyio
async def test_forgot_password_always_200():
    """Всегда 200 — не раскрывает наличие email."""
    async with _make_client() as ac:
        r = await ac.post("/api/auth/forgot-password", json={"email": "ghost@nowhere.com"})
    assert r.status_code == 200
    assert "message" in r.json()


@pytest.mark.anyio
async def test_forgot_password_creates_token():
    """Для существующего пользователя создаётся токен в _reset_tokens."""
    async with _make_client() as ac:
        await _register(ac, "reset@test.com", "pass123")
        r = await ac.post("/api/auth/forgot-password", json={"email": "reset@test.com"})
    assert r.status_code == 200
    # Должен быть хотя бы один токен с этим email
    found = any(v["email"] == "reset@test.com" for v in _reset_tokens.values())
    assert found, "Токен сброса не был создан"


@pytest.mark.anyio
async def test_forgot_password_rate_limit():
    """Rate limit применяется и к forgot-password."""
    from app.main import _AUTH_LIMIT_PER_MINUTE
    import time
    _auth_attempts["testclient"] = {"count": _AUTH_LIMIT_PER_MINUTE, "window_start": time.time()}
    async with _make_client() as ac:
        r = await ac.post("/api/auth/forgot-password", json={"email": "any@test.com"})
    assert r.status_code == 429


# ─── /api/auth/reset-password (P35) ──────────────────────────────────────────

@pytest.mark.anyio
async def test_reset_password_invalid_token():
    async with _make_client() as ac:
        r = await ac.post(
            "/api/auth/reset-password",
            json={"token": "invalid-token-xyz", "new_password": "newpass123"},
        )
    assert r.status_code == 400
    assert "недействительна" in r.json()["detail"].lower() or "invalid" in r.json()["detail"].lower()


@pytest.mark.anyio
async def test_reset_password_expired_token():
    """Просроченный токен — 400."""
    import time
    _reset_tokens["expired-token"] = {"email": "test@test.com", "exp": time.time() - 1}
    async with _make_client() as ac:
        r = await ac.post(
            "/api/auth/reset-password",
            json={"token": "expired-token", "new_password": "newpass123"},
        )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_reset_password_full_flow():
    """Полный сценарий: регистрация → forgot → reset → вход с новым паролем."""
    import time, secrets
    email = "fullreset@test.com"
    async with _make_client() as ac:
        await _register(ac, email, "oldpassword")

        # Запрашиваем сброс
        r = await ac.post("/api/auth/forgot-password", json={"email": email})
        assert r.status_code == 200

        # Находим токен
        token = next(
            (t for t, d in _reset_tokens.items() if d["email"] == email),
            None,
        )
        assert token is not None, "Токен не создан"

        # Сбрасываем пароль
        r2 = await ac.post(
            "/api/auth/reset-password",
            json={"token": token, "new_password": "brandnewpass"},
        )
        assert r2.status_code == 200

        # Токен должен быть удалён
        assert token not in _reset_tokens

        # Старый пароль больше не работает
        r3 = await ac.post("/api/auth/login", json={"email": email, "password": "oldpassword"})
        assert r3.status_code == 401

        # Новый пароль работает
        r4 = await ac.post("/api/auth/login", json={"email": email, "password": "brandnewpass"})
        assert r4.status_code == 200


@pytest.mark.anyio
async def test_reset_password_token_single_use():
    """Токен можно использовать только один раз."""
    import time
    email = "singleuse@test.com"
    async with _make_client() as ac:
        await _register(ac, email, "pass123")
        await ac.post("/api/auth/forgot-password", json={"email": email})
        token = next(t for t, d in _reset_tokens.items() if d["email"] == email)

        # Первый сброс — успешен
        r1 = await ac.post(
            "/api/auth/reset-password",
            json={"token": token, "new_password": "newpass1"},
        )
        assert r1.status_code == 200

        # Второй — токен уже удалён
        r2 = await ac.post(
            "/api/auth/reset-password",
            json={"token": token, "new_password": "newpass2"},
        )
        assert r2.status_code == 400


# ─── /api/auth/history (P31) ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_history_requires_auth():
    async with _make_client() as ac:
        r = await ac.get("/api/auth/history")
    assert r.status_code == 401


@pytest.mark.anyio
async def test_history_initially_empty():
    async with _make_client() as ac:
        token = await _register(ac, "hist@test.com")
        r = await ac.get("/api/auth/history", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    d = r.json()
    assert "records" in d
    assert "stats" in d
    assert isinstance(d["records"], list)
    assert len(d["records"]) == 0


@pytest.mark.anyio
async def test_history_record_and_delete():
    """Создаём запись истории, затем удаляем."""
    async with _make_client() as ac:
        token = await _register(ac, "histdel@test.com")
        headers = {"Authorization": f"Bearer {token}"}

        # Добавляем запись
        r = await ac.post(
            "/api/auth/record",
            json={"filename": "track.wav", "style": "standard", "out_format": "wav",
                  "before_lufs": -18.5, "after_lufs": -14.0},
            headers=headers,
        )
        assert r.status_code == 200
        rec_id = r.json()["id"]

        # Проверяем что появилась
        r2 = await ac.get("/api/auth/history", headers=headers)
        assert r2.status_code == 200
        records = r2.json()["records"]
        assert len(records) == 1
        assert records[0]["filename"] == "track.wav"

        # Удаляем
        r3 = await ac.delete(f"/api/auth/history/{rec_id}", headers=headers)
        assert r3.status_code == 200

        # Проверяем что удалена
        r4 = await ac.get("/api/auth/history", headers=headers)
        assert len(r4.json()["records"]) == 0


# ─── /api/auth/logout ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_logout_always_200():
    async with _make_client() as ac:
        r = await ac.post("/api/auth/logout")
    assert r.status_code == 200
    assert "message" in r.json()
