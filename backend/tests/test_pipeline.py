"""Тесты для pipeline.py: цепочка мастеринга, анализ, экспорт.

Запуск: cd backend && python -m pytest tests/test_pipeline.py -v
"""
import io
import pytest
import numpy as np

# Минимальная длина для pyloudnorm (integrated_loudness)
MIN_LUFS_SAMPLES = 88200  # ~2 сек при 44100 Hz


@pytest.fixture
def stereo_2sec_44k():
    """Стерео сигнал 2 с, 44.1 kHz, тихий тон (для стабильного LUFS)."""
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, dtype=np.float32)
    # Синус ~400 Hz, -20 dB примерная громкость
    left = (0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    right = (0.1 * np.sin(2 * np.pi * 400 * t + 0.1)).astype(np.float32)
    return np.column_stack((left, right)), sr


@pytest.fixture
def mono_2sec_44k():
    """Моно сигнал 2 с, 44.1 kHz."""
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, dtype=np.float32)
    sig = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return sig, sr


def test_remove_dc_offset_mono():
    from app.pipeline import remove_dc_offset
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = remove_dc_offset(x)
    assert out.dtype == np.float32
    assert np.isclose(np.mean(out), 0.0)


def test_remove_dc_offset_stereo():
    from app.pipeline import remove_dc_offset
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    out = remove_dc_offset(x)
    assert out.shape == x.shape
    assert np.isclose(np.mean(out[:, 0]), 0.0)
    assert np.isclose(np.mean(out[:, 1]), 0.0)


def test_measure_lufs_silence_returns_finite_or_nan():
    from app.pipeline import measure_lufs
    sr = 44100
    silence = np.zeros((sr * 2, 2), dtype=np.float32)
    result = measure_lufs(silence, sr)
    assert isinstance(result, float)
    # Тишина: -inf или nan
    assert result != result or result <= -50  # nan or very quiet


def test_measure_lufs_signal(stereo_2sec_44k):
    from app.pipeline import measure_lufs
    audio, sr = stereo_2sec_44k
    lufs = measure_lufs(audio, sr)
    assert isinstance(lufs, float)
    assert not np.isnan(lufs)
    assert -60 < lufs < 0


def test_compute_spectrum_bars_returns_64_bars(stereo_2sec_44k):
    from app.pipeline import compute_spectrum_bars
    audio, sr = stereo_2sec_44k
    bars = compute_spectrum_bars(audio, sr, n_bars=64)
    assert len(bars) == 64
    assert all(isinstance(b, (int, float)) for b in bars)


def test_compute_spectrum_bars_short_audio():
    from app.pipeline import compute_spectrum_bars
    short = np.zeros(1000, dtype=np.float32)
    bars = compute_spectrum_bars(short, 44100, n_fft=4096, n_bars=64)
    assert len(bars) == 64
    assert all(b <= 0 for b in bars)


def test_compute_vectorscope_points_mono_returns_empty():
    from app.pipeline import compute_vectorscope_points
    mono = np.zeros((1000, 1), dtype=np.float32)
    points = compute_vectorscope_points(mono)
    assert points == []


def test_compute_vectorscope_points_stereo_returns_points(stereo_2sec_44k):
    from app.pipeline import compute_vectorscope_points
    audio, _ = stereo_2sec_44k
    points = compute_vectorscope_points(audio, max_points=100)
    assert len(points) <= 100
    assert len(points) > 0
    for p in points:
        assert len(p) == 2
        assert -1.01 <= p[0] <= 1.01 and -1.01 <= p[1] <= 1.01


def test_compute_lufs_timeline_short_audio():
    from app.pipeline import compute_lufs_timeline
    sr = 44100
    short = np.zeros(sr // 2, dtype=np.float32)  # 0.5 сек
    timeline, step = compute_lufs_timeline(short, sr, block_sec=0.4, max_points=300)
    assert isinstance(timeline, list)
    assert len(timeline) >= 1
    assert isinstance(step, (int, float))


def test_compute_lufs_timeline_longer_audio(stereo_2sec_44k):
    from app.pipeline import compute_lufs_timeline
    audio, sr = stereo_2sec_44k
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    timeline, step = compute_lufs_timeline(mono, sr, block_sec=0.4, max_points=50)
    assert len(timeline) >= 1
    assert step >= 0


def test_measure_stereo_correlation_mono_returns_none():
    from app.pipeline import measure_stereo_correlation
    mono = np.zeros((1000, 1), dtype=np.float32)
    assert measure_stereo_correlation(mono) is None


def test_measure_stereo_correlation_stereo_in_range(stereo_2sec_44k):
    from app.pipeline import measure_stereo_correlation
    audio, _ = stereo_2sec_44k
    corr = measure_stereo_correlation(audio)
    assert corr is not None
    assert -1.01 <= corr <= 1.01


def test_export_audio_wav_returns_bytes(stereo_2sec_44k):
    from app.pipeline import export_audio
    audio, sr = stereo_2sec_44k
    out = export_audio(audio, sr, channels=2, out_format="wav", dither_type="tpdf")
    assert isinstance(out, bytes)
    assert len(out) > 100
    assert out[:4] == b"RIFF"


def test_export_audio_wav_dither_types(stereo_2sec_44k):
    from app.pipeline import export_audio
    audio, sr = stereo_2sec_44k
    for dither in ("tpdf", "ns_e"):
        out = export_audio(audio, sr, channels=2, out_format="wav", dither_type=dither)
        assert isinstance(out, bytes)
        assert out[:4] == b"RIFF"


def test_export_audio_aac_returns_bytes(stereo_2sec_44k):
    import shutil
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for AAC export")
    from app.pipeline import export_audio
    audio, sr = stereo_2sec_44k
    out = export_audio(audio, sr, channels=2, out_format="aac")
    assert isinstance(out, bytes)
    assert len(out) > 100
    # M4A (ipod) container: first 4 bytes size, then 'ftyp'
    assert b"ftyp" in out[:20]


def test_run_mastering_pipeline_returns_same_shape(stereo_2sec_44k):
    from app.pipeline import run_mastering_pipeline
    audio, sr = stereo_2sec_44k
    progress_log = []

    def progress(pct: int, msg: str):
        progress_log.append((pct, msg))

    out = run_mastering_pipeline(
        audio,
        sr,
        target_lufs=-14.0,
        style="standard",
        progress_callback=progress,
    )
    assert out.shape == audio.shape
    assert out.dtype == np.float32
    assert not np.any(np.isnan(out))
    assert np.all(np.abs(out) <= 1.01)
    assert len(progress_log) >= 5


def test_run_mastering_pipeline_style_edm(stereo_2sec_44k):
    from app.pipeline import run_mastering_pipeline
    audio, sr = stereo_2sec_44k
    out = run_mastering_pipeline(audio, sr, target_lufs=-9.0, style="edm")
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))


def test_load_audio_from_bytes_wav():
    from app.pipeline import load_audio_from_bytes
    import soundfile as sf
    buf = io.BytesIO()
    samples = np.zeros((44100, 2), dtype=np.float32)
    sf.write(buf, samples, 44100, format="WAV", subtype="FLOAT")
    data = buf.getvalue()
    loaded, sr = load_audio_from_bytes(data, "test.wav")
    assert sr == 44100
    assert loaded.ndim == 2
    assert loaded.shape[1] == 2


def test_style_configs_has_required_styles():
    from app.pipeline import STYLE_CONFIGS
    required = {"standard", "edm", "hiphop", "classical", "podcast", "lofi", "house_basic"}
    for style in required:
        assert style in STYLE_CONFIGS
        cfg = STYLE_CONFIGS[style]
        assert "lufs" in cfg
        assert isinstance(cfg["lufs"], (int, float))


def test_denoise_presets():
    from app.pipeline import DENOISE_PRESETS, apply_spectral_denoise
    assert set(DENOISE_PRESETS) == {"light", "medium", "aggressive"}
    for name, (strength, noise_pct) in DENOISE_PRESETS.items():
        assert 0 < strength <= 1
        assert 5 <= noise_pct <= 30
    # apply_spectral_denoise с пресетом medium не ломает сигнал
    audio = np.zeros((2048 * 4, 2), dtype=np.float32)
    audio[:, 0] = 0.1 * np.sin(np.linspace(0, 100, audio.shape[0]))
    audio[:, 1] = audio[:, 0] * 0.9
    out = apply_spectral_denoise(audio, 44100, strength=0.5, noise_percentile=15.0)
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))
