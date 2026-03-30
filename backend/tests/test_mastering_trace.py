"""Трассировка мастеринга (MAGIC_MASTER_MASTERING_TRACE)."""
import logging

import numpy as np
import pytest


@pytest.fixture
def mono_2sec_44k():
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, dtype=np.float32)
    sig = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return sig, sr


def test_trace_stage_logs_when_env_enabled(caplog, monkeypatch):
    monkeypatch.setenv("MAGIC_MASTER_MASTERING_TRACE", "1")
    caplog.set_level(logging.INFO, logger="app.mastering_trace")

    from app.mastering_trace import TraceContext, trace_stage

    ctx = TraceContext.build("job-abc", "track.wav", "v1", target_lufs=-14.0, style="standard")
    trace_stage(ctx, "unit_test_stage", np.ones(500, dtype=np.float32) * 0.01, 44100)

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "mastering_trace" in joined
    assert "job-abc" in joined
    assert "unit_test_stage" in joined


def test_run_mastering_pipeline_emits_trace_stages(caplog, monkeypatch, mono_2sec_44k):
    monkeypatch.setenv("MAGIC_MASTER_MASTERING_TRACE", "1")
    caplog.set_level(logging.INFO, logger="app.mastering_trace")

    from app.mastering_trace import TraceContext
    from app.pipeline import run_mastering_pipeline

    audio, sr = mono_2sec_44k
    ctx = TraceContext.build("t-pipe", "in.wav", "v1", target_lufs=-14.0, style="standard")
    run_mastering_pipeline(
        audio,
        sr,
        target_lufs=-14.0,
        style="standard",
        trace_ctx=ctx,
    )

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "mastering_trace" in joined
    assert "dc_offset" in joined
    assert "t-pipe" in joined
