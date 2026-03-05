"""API tests for Magic Master — FastAPI backend.

Run:
    cd backend && python3 -m pytest tests/ -v
"""
import sys
import os
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app
from app.deps import _rate_limits
from app.helpers import check_audio_magic_bytes as _check_audio_magic_bytes


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
async def test_api_health_returns_features():
    """GET /api/health возвращает features (ai_enabled, batch_enabled, vocal_isolation_enabled и др.) для фронта."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert "features" in data
    feats = data["features"]
    assert "vocal_isolation_enabled" in feats
    assert isinstance(feats["vocal_isolation_enabled"], bool)
    for key in ("ai_enabled", "batch_enabled", "registration_enabled"):
        assert key in feats
        assert isinstance(feats[key], bool)


@pytest.mark.anyio
async def test_api_metrics():
    """P58: эндпоинт метрик для внешнего мониторинга."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "uptime_seconds" in data
    assert "jobs_running" in data
    assert "jobs_total" in data
    assert "version" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0
    assert isinstance(data["jobs_running"], int)
    assert isinstance(data["jobs_total"], int)


@pytest.mark.anyio
async def test_api_locale():
    """P59: эндпоинт доступных локалей (i18n)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/locale")
    assert r.status_code == 200
    data = r.json()
    assert "available" in data
    assert "default" in data
    assert "ru" in data["available"]
    assert "en" in data["available"]
    assert data["default"] == "ru"


def test_check_audio_magic_bytes():
    """P60: проверка magic bytes для загружаемых файлов."""
    # WAV: RIFF....WAVE
    assert _check_audio_magic_bytes(b"RIFF\x00\x00\x00\x00WAVE\x00\x00", "a.wav") is True
    assert _check_audio_magic_bytes(b"RIFF\x10\x00\x00\x00WAVE", "x.wav") is True
    assert _check_audio_magic_bytes(b"fLaC\x00", "a.wav") is False
    # FLAC
    assert _check_audio_magic_bytes(b"fLaC\x00\x00\x00", "a.flac") is True
    assert _check_audio_magic_bytes(b"RIFF", "a.flac") is False
    # MP3: ID3 или 0xFF 0xE?
    assert _check_audio_magic_bytes(b"ID3\x04", "a.mp3") is True
    assert _check_audio_magic_bytes(b"\xff\xfb\x90\x00", "a.mp3") is True
    assert _check_audio_magic_bytes(b"\xff\xfa\x00\x00", "a.mp3") is True
    assert _check_audio_magic_bytes(b"RIFF", "a.mp3") is False
    # неизвестное расширение — пропуск проверки
    assert _check_audio_magic_bytes(b"xxx", "a.xyz") is True
    # пустые данные
    assert _check_audio_magic_bytes(b"", "a.wav") is True


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
async def test_api_presets_community():
    """P64: пресеты сообщества."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/presets/community")
    assert r.status_code == 200
    data = r.json()
    assert "presets" in data
    assert isinstance(data["presets"], list)
    for p in data["presets"]:
        assert "id" in p and "name" in p
        assert "target_lufs" in p


@pytest.mark.anyio
async def test_api_extensions():
    """Минимальный API расширений: статус загрузки дополнительных пресетов."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/extensions")
    assert r.status_code == 200
    data = r.json()
    assert "community_presets_extra_configured" in data
    assert "community_presets_extra_loaded" in data
    assert isinstance(data["community_presets_extra_configured"], bool)
    assert isinstance(data["community_presets_extra_loaded"], bool)


@pytest.mark.anyio
async def test_api_styles():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/styles")
    assert r.status_code == 200
    data = r.json()
    assert "styles" in data
    assert "standard" in data["styles"]


@pytest.mark.anyio
async def test_api_styles_each_has_lufs():
    """GET /api/styles: у каждого стиля есть ключ lufs и значение — число."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/styles")
    assert r.status_code == 200
    data = r.json()
    assert "styles" in data
    for name, style in data["styles"].items():
        assert isinstance(style, dict), f"style {name!r} must be dict"
        assert "lufs" in style, f"style {name!r} missing lufs"
        assert isinstance(style["lufs"], (int, float)), f"style {name!r}.lufs must be number"


@pytest.mark.anyio
async def test_api_v2_chain_default():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/v2/chain/default", params={"style": "standard", "target_lufs": -14})
    assert r.status_code == 200
    data = r.json()
    assert data.get("version") == "v2"
    assert data.get("style") == "standard"
    assert data.get("target_lufs") == -14.0
    assert "modules" in data
    assert len(data["modules"]) > 0
    for m in data["modules"]:
        assert "id" in m
        assert "label" in m
        assert isinstance(m["label"], str)


@pytest.mark.anyio
async def test_api_v2_chain_default_style_dry_vocal():
    """GET /api/v2/chain/default?style=dry_vocal возвращает цепочку для стиля dry_vocal."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/v2/chain/default", params={"style": "dry_vocal", "target_lufs": -14})
    assert r.status_code == 200
    data = r.json()
    assert data.get("style") == "dry_vocal"
    assert data.get("target_lufs") == -14.0
    assert "modules" in data
    assert len(data["modules"]) > 0


@pytest.mark.anyio
async def test_api_v2_master_unknown_style_fallback(minimal_wav_bytes):
    """POST /api/v2/master с неизвестным style принимает запрос (бэкенд подставляет standard)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"style": "unknown_style_xyz", "out_format": "wav"},
        )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "running"


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
async def test_api_measure_returns_lufs(minimal_wav_bytes):
    """POST /api/measure возвращает lufs, sample_rate, peak_dbfs, duration, channels."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/measure",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
        )
    assert r.status_code == 200
    data = r.json()
    for key in ("lufs", "sample_rate", "peak_dbfs", "duration", "channels"):
        assert key in data, f"Missing field: {key}"
    assert isinstance(data["lufs"], (int, float))
    assert data["sample_rate"] == 44100
    assert data["channels"] == 1
    assert isinstance(data["duration"], (int, float))
    assert data["duration"] >= 0


@pytest.mark.anyio
async def test_api_measure_rejects_bad_extension():
    """POST /api/measure с неподдерживаемым расширением возвращает 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/measure",
            files={"file": ("bad.txt", b"not audio", "text/plain")},
        )
    assert r.status_code == 400


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
async def test_api_v2_analyze_extended(minimal_wav_bytes):
    """POST /api/v2/analyze с extended=true может вернуть spectrum_bars, lufs_timeline."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/analyze",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"extended": "true"},
        )
    assert r.status_code == 200
    data = r.json()
    assert "lufs" in data and "duration_sec" in data
    # При extended могут быть spectrum_bars и/или lufs_timeline (зависит от длины аудио)
    if "spectrum_bars" in data:
        assert isinstance(data["spectrum_bars"], list)
    if "lufs_timeline" in data:
        assert isinstance(data["lufs_timeline"], list)


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
    from app.deps import _FREE_DAILY_LIMIT
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
async def test_api_v2_master_accepts_bitrate(minimal_wav_bytes):
    """POST /api/v2/master принимает параметр bitrate (для MP3/OPUS); при WAV битрейт игнорируется."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"style": "standard", "out_format": "wav", "bitrate": "320"},
        )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "running"


@pytest.mark.anyio
async def test_api_v2_master_accepts_pro_params(minimal_wav_bytes):
    """POST /api/v2/master принимает PRO-параметры (rumble, denoiser, deesser, transient, parallel, dynamic_eq)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={
                "style": "standard",
                "out_format": "wav",
                "rumble_enabled": "true",
                "rumble_cutoff": "80",
                "denoise_preset": "light",
                "deesser_enabled": "true",
                "deesser_threshold": "-6",
                "deesser_freq_hi": "9000",
                "transient_attack": "1.1",
                "transient_sustain": "0.9",
                "parallel_mix": "0.3",
                "dynamic_eq_enabled": "true",
            },
        )
    assert r.status_code == 200, (r.status_code, r.text)
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "running"


@pytest.mark.anyio
async def test_api_v2_batch_requires_files():
    """POST /api/v2/batch без файлов или с пустым списком — 422 (валидация) или 400/503 (бизнес-логика)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Без поля files — FastAPI вернёт 422
        r = await ac.post(
            "/api/v2/batch",
            data={"style": "standard", "out_format": "wav"},
        )
    assert r.status_code in (400, 422, 503)


@pytest.mark.anyio
async def test_api_v2_batch_with_one_file_creates_job(minimal_wav_bytes):
    """POST /api/v2/batch с одним валидным файлом возвращает 200 и список jobs с job_id (при включённой фиче batch)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/batch",
            data={"style": "standard", "out_format": "wav"},
            files=[("files", ("one.wav", minimal_wav_bytes, "audio/wav"))],
        )
    if r.status_code == 503:
        detail = (r.json().get("detail") or r.text) or ""
        if "batch" in detail.lower() or "пакетн" in detail.lower():
            pytest.skip("Фича пакетной обработки отключена на сервере")
    assert r.status_code == 200, (r.status_code, r.text)
    data = r.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)
    assert len(data["jobs"]) == 1
    assert "job_id" in data["jobs"][0]
    assert "filename" in data["jobs"][0]


@pytest.mark.anyio
async def test_api_master_status_structure(minimal_wav_bytes):
    """GET /api/master/status/{job_id} возвращает status, progress, message (и опционально error)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        create = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={"style": "standard", "out_format": "wav"},
        )
    assert create.status_code == 200
    job_id = create.json().get("job_id")
    assert job_id
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get(f"/api/master/status/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] in ("running", "done", "error")
    assert "progress" in data
    assert isinstance(data["progress"], int)
    assert 0 <= data["progress"] <= 100
    assert "message" in data
    assert isinstance(data["message"], str)


@pytest.mark.anyio
async def test_api_master_status_404():
    """GET /api/master/status/{job_id} с несуществующим id → 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/master/status/nonexistent-job-id")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_api_master_result_404():
    """GET /api/master/result/{job_id} с несуществующим id → 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/master/result/nonexistent-job-id")
    assert r.status_code == 404


# ─── Изоляция вокала (9.2) ───────────────────────────────────────────────────

@pytest.mark.anyio
async def test_api_v2_isolate_vocal_disabled(minimal_wav_bytes):
    """POST /api/v2/isolate-vocal при выключенной фиче (по умолчанию) → 503."""
    from app.config import settings
    if getattr(settings, "enable_vocal_isolation", False):
        pytest.skip("Фича изоляции вокала включена в конфиге; тест проверяет ответ при выключенной фиче")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/isolate-vocal",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
        )
    assert r.status_code == 503
    detail = r.json().get("detail", r.text)
    assert "отключена" in detail or "ENABLE_VOCAL_ISOLATION" in detail or "demucs" in detail.lower()


@pytest.mark.anyio
async def test_api_v2_isolate_vocal_rejects_bad_extension():
    """POST /api/v2/isolate-vocal с неподдерживаемым форматом → 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post(
            "/api/v2/isolate-vocal",
            files={"file": ("document.txt", b"not audio content", "text/plain")},
        )
    assert r.status_code == 400
