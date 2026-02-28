"""
E2E-тест полного цикла мастеринга: загрузка файла → POST /api/v2/master → опрос статуса → скачивание результата.
P63. Запуск: cd backend && PYTHONPATH=. python3 -m pytest tests/test_e2e_mastering.py -v
"""
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.anyio
async def test_e2e_mastering_flow(minimal_wav_bytes: bytes):
    """
    Полный сценарий: отправить WAV → получить job_id → дождаться done → скачать результат (WAV).
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=60.0,
    ) as ac:
        # 1) Запуск мастеринга
        r = await ac.post(
            "/api/v2/master",
            files={"file": ("test.wav", minimal_wav_bytes, "audio/wav")},
            data={
                "target_lufs": "-14",
                "style": "standard",
                "out_format": "wav",
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "job_id" in data
        job_id = data["job_id"]
        assert "status" in data

        # 2) Ожидание завершения (опрос статуса)
        for _ in range(60):
            r_status = await ac.get(f"/api/master/status/{job_id}")
            assert r_status.status_code == 200
            st = r_status.json()
            status = st.get("status")
            if status == "done":
                break
            if status == "error":
                pytest.fail(f"Mastering failed: {st.get('error', 'unknown')}")
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Timeout waiting for job to complete")

        # 3) Скачивание результата
        r_result = await ac.get(f"/api/master/result/{job_id}")
        assert r_result.status_code == 200
        body = r_result.content
        assert len(body) > 100
        assert body[:4] == b"RIFF", "Expected WAV output"
        assert b"WAVE" in body[:20]
