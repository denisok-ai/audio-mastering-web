"""API tests for Magic Master — FastAPI backend.

Run:
    cd backend && python3 -m pytest tests/ -v
"""
import pytest
from httpx import AsyncClient, ASGITransport

# Reset rate limits before each test to avoid interference
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app, _rate_limits


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Сбрасываем счётчики лимитов между тестами."""
    _rate_limits.clear()
    yield
    _rate_limits.clear()


@pytest.mark.anyio
async def test_api_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api")
    assert r.status_code == 200
    data = r.json()
    assert "service" in data
    assert "version" in data


@pytest.mark.anyio
async def test_api_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_api_version():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


@pytest.mark.anyio
async def test_api_progress():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/progress")
    assert r.status_code == 200
    assert "markdown" in r.headers.get("content-type", "")


@pytest.mark.anyio
async def test_api_presets():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/presets")
    assert r.status_code == 200
    data = r.json()
    assert "presets" in data
    assert isinstance(data["presets"], dict)
    assert len(data["presets"]) > 0


@pytest.mark.anyio
async def test_api_styles():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/styles")
    assert r.status_code == 200
    data = r.json()
    assert "styles" in data
    assert "standard" in data["styles"]


@pytest.mark.anyio
async def test_api_v2_chain_default():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/v2/chain/default", params={"style": "standard", "target_lufs": -14})
    assert r.status_code == 200
    data = r.json()
    assert "modules" in data
    assert len(data["modules"]) > 0
    assert all("id" in m for m in data["modules"])


# ─── Rate limits ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_api_limits_default():
    """Без использований — remaining == limit."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/limits")
    assert r.status_code == 200
    data = r.json()
    assert data["tier"] == "free"
    assert data["remaining"] == data["limit"]
    assert data["used"] == 0
    assert "reset_at" in data


@pytest.mark.anyio
async def test_api_limits_fields():
    """Ответ содержит все нужные поля."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/limits")
    data = r.json()
    for field in ("tier", "used", "limit", "remaining", "reset_at"):
        assert field in data, f"Missing field: {field}"


@pytest.mark.anyio
async def test_api_analyze_rejects_bad_extension():
    """POST /api/v2/analyze отклоняет неверное расширение."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/analyze",
            files={"file": ("bad.txt", b"not audio", "text/plain")},
            data={"extended": "false"},
        )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_api_v2_analyze_basic(minimal_wav_bytes):
    """POST /api/v2/analyze возвращает основные поля."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/analyze",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"extended": "false"},
        )
    assert r.status_code == 200
    data = r.json()
    for field in ("lufs", "duration_sec", "sample_rate", "channels"):
        assert field in data, f"Missing field: {field}"
    assert data["sample_rate"] == 44100
    assert data["channels"] == 1


@pytest.mark.anyio
async def test_api_v2_master_creates_job(minimal_wav_bytes):
    """POST /api/v2/master создаёт задачу и возвращает job_id."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"style": "standard", "out_format": "wav"},
        )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "running"


@pytest.mark.anyio
async def test_api_v2_master_rate_limit(minimal_wav_bytes):
    """После 3 запросов с одного IP — должен вернуть 429."""
    from app.main import _FREE_DAILY_LIMIT
    import datetime

    # Заполняем лимит напрямую (имитируем 3 успешных мастеринга)
    today = datetime.date.today().isoformat()
    _rate_limits["testclient"] = {"count": _FREE_DAILY_LIMIT, "day": today}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"style": "standard", "out_format": "wav"},
        )
    assert r.status_code == 429
    detail = r.json().get("detail", "")
    assert "лимит" in detail.lower() or "limit" in detail.lower() or "429" in str(r.status_code)


@pytest.mark.anyio
async def test_api_master_status_404():
    """GET /api/master/status/{job_id} с несуществующим id → 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/master/status/nonexistent-job-id")
    assert r.status_code == 404
