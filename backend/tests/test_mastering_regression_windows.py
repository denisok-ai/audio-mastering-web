"""Регрессия мастеринга по окнам времени (фикстура WAV или синтетика)."""
import os

import numpy as np
import pytest

from app.qa.mastering_regression import (
    DEFAULT_WINDOWS_SEC,
    hf_rms,
    load_expected_thresholds,
    metrics_after_each_stage,
    regression_wav_path,
    run_default_chain_stages,
    slice_window,
    to_mono_float64,
    window_metrics,
)


@pytest.fixture
def windows_short_track():
    """Короткий «трек» ~48 с: окна в пределах длины."""
    return (
        ("intro", 2.0, 10.0),
        ("mid", 20.0, 30.0),
        ("late", 40.0, 48.0),
    )


def test_synthetic_noise_mastering_finite_and_moderate_hf(windows_short_track):
    """~48 с шум: цепочка не даёт NaN; ВЧ-энергия на выходе не взрывается относительно входа."""
    rng = np.random.default_rng(42)
    sr = 48000
    dur = 48.0
    n = int(sr * dur)
    x = (0.04 * rng.standard_normal(n)).astype(np.float32)
    mono_in = to_mono_float64(x)
    wdef = windows_short_track
    in_metrics = window_metrics(x, sr, wdef)
    stages = run_default_chain_stages(x, sr, target_lufs=-14.0, style="standard")
    assert len(stages) >= 3
    final_id, final_buf = stages[-1]
    assert final_id == "v2_output_fade_in"
    assert np.all(np.isfinite(final_buf))
    out_metrics = window_metrics(final_buf, sr, wdef)
    for name in in_metrics:
        ratio = out_metrics[name]["hf_rms"] / (in_metrics[name]["hf_rms"] + 1e-12)
        assert ratio < 80.0, f"{name} hf blowup ratio={ratio}"
        assert out_metrics[name]["max_abs_diff"] < 1.5


@pytest.mark.skipif(regression_wav_path() is None, reason="Нет MM_REGRESSION_WAV и нет fixtures/alors_on_danse_rem.wav")
def test_fixture_regression_windows_vs_input():
    path = regression_wav_path()
    assert path is not None
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.shape[1] >= 2:
        audio = np.mean(data, axis=1).astype(np.float32)
    else:
        audio = data[:, 0].astype(np.float32)
    min_sec = max(t1 for _, _, t1 in DEFAULT_WINDOWS_SEC) + 0.5
    assert len(audio) / sr >= min_sec, f"Файл короче {min_sec}s для окон DEFAULT_WINDOWS_SEC"

    mono_in = to_mono_float64(audio)
    in_win = {n: slice_window(mono_in, sr, t0, t1) for n, t0, t1 in DEFAULT_WINDOWS_SEC}
    in_hf = {n: hf_rms(w, sr) for n, w in in_win.items()}

    stages = run_default_chain_stages(audio, sr, target_lufs=-14.0, style="standard")
    _, final_buf = stages[-1]
    assert np.all(np.isfinite(final_buf))
    mono_out = to_mono_float64(final_buf)
    exp = load_expected_thresholds()
    default_ratio = 35.0
    default_diff = 0.95
    for name, t0, t1 in DEFAULT_WINDOWS_SEC:
        wo = slice_window(mono_out, sr, t0, t1)
        wi = in_win[name]
        out_hf = hf_rms(wo, sr)
        ratio = out_hf / (in_hf[name] + 1e-12)
        max_r = default_ratio
        if exp and "max_hf_rms_ratio_vs_input" in exp:
            max_r = float(exp["max_hf_rms_ratio_vs_input"].get(name, max_r))
        assert ratio < max_r, f"{name}: hf ratio {ratio} >= {max_r}"

        md = float(np.max(np.abs(np.diff(wo))) if wo.size > 1 else 0.0)
        max_d = default_diff
        if exp and "max_final_abs_diff" in exp:
            max_d = float(exp["max_final_abs_diff"])
        assert md < max_d, f"{name}: max_abs_diff {md} >= {max_d}"


def test_metrics_after_each_stage_keys():
    rng = np.random.default_rng(1)
    sr = 44100
    n = sr * 6
    x = (0.05 * rng.standard_normal(n)).astype(np.float32)
    rows = metrics_after_each_stage(x, sr, windows_sec=(("a", 1.0, 3.0),))
    assert len(rows) >= 2
    assert rows[0]["stage"] == "dc_offset"
    assert "a" in rows[0]["windows"]
