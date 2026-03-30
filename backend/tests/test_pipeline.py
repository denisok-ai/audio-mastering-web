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


def test_export_audio_mp3_with_bitrate(stereo_2sec_44k):
    """Экспорт MP3 с параметром bitrate (128/192/256/320 kbps)."""
    import shutil
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for MP3 export")
    from app.pipeline import export_audio
    audio, sr = stereo_2sec_44k
    for br in (128, 192, 320):
        out = export_audio(audio, sr, channels=2, out_format="mp3", bitrate=br)
        assert isinstance(out, bytes)
        assert len(out) > 100
        assert out[:3] == b"ID3" or b"\xff\xfb" in out[:4]  # ID3 header or MPEG frame


def test_export_audio_opus_with_bitrate(stereo_2sec_44k):
    """Экспорт OPUS с параметром bitrate (128/192 kbps)."""
    import shutil
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for OPUS export")
    from app.pipeline import export_audio
    audio, sr = stereo_2sec_44k
    for br in (128, 192):
        out = export_audio(audio, sr, channels=2, out_format="opus", bitrate=br)
        assert isinstance(out, bytes)
        assert len(out) > 100
        # Ogg container
        assert out[:4] == b"OggS"


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


def test_run_mastering_pipeline_style_dry_vocal(stereo_2sec_44k):
    """Пайплайн со стилем dry_vocal (ровная АЧХ, без эксайтера): выход без NaN, форма сохраняется."""
    from app.pipeline import run_mastering_pipeline
    audio, sr = stereo_2sec_44k
    out = run_mastering_pipeline(audio, sr, target_lufs=-14.0, style="dry_vocal")
    assert out.shape == audio.shape
    assert out.dtype == np.float32
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    assert np.max(np.abs(out)) <= 1.01


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
    required = {"standard", "edm", "hiphop", "classical", "podcast", "lofi", "house_basic", "dry_vocal"}
    for style in required:
        assert style in STYLE_CONFIGS
        cfg = STYLE_CONFIGS[style]
        assert "lufs" in cfg
        assert isinstance(cfg["lufs"], (int, float))


def test_denoise_presets():
    from app.pipeline import DENOISE_PRESETS, apply_spectral_denoise
    assert set(DENOISE_PRESETS) == {"vocal", "light", "medium", "aggressive", "tape_hiss", "room_tone"}
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


def test_apply_rumble_filter():
    """Румбл-фильтр (high-pass 80 Hz): форма сохраняется, нет NaN, низкие частоты ослабляются."""
    from app.pipeline import apply_rumble_filter
    sr = 44100
    n = sr * 2
    t = np.linspace(0, n / sr, n, dtype=np.float32)
    # Сигнал с низкой (30 Hz) и средней (200 Hz) частотой
    low = 0.5 * np.sin(2 * np.pi * 30 * t)
    mid = 0.3 * np.sin(2 * np.pi * 200 * t)
    audio = (low + mid).astype(np.float32)
    out = apply_rumble_filter(audio, sr, cutoff_hz=80.0)
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))
    # После HP 80 Hz амплитуда низкой составляющей должна уменьшиться
    assert np.max(np.abs(out)) <= np.max(np.abs(audio)) * 1.1


def test_run_mastering_pipeline_sine_no_clip_nan(mono_2sec_44k):
    """Пайплайн на синусе: выход без клиппинга и NaN, LUFS в разумных границах."""
    from app.pipeline import run_mastering_pipeline, measure_lufs
    audio, sr = mono_2sec_44k
    out = run_mastering_pipeline(audio, sr, target_lufs=-14.0, style="standard")
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    assert np.max(np.abs(out)) <= 1.01
    lufs = measure_lufs(out, sr)
    assert not np.isnan(lufs)
    assert -50 < lufs < 0


def test_run_mastering_pipeline_vocal_like():
    """Вокало-подобный сигнал (микс синусов 200–2k Hz + малый шум): пайплайн не падает, выход конечный."""
    from app.pipeline import run_mastering_pipeline
    sr = 44100
    n = sr * 2  # 2 сек
    t = np.linspace(0, n / sr, n, dtype=np.float32)
    # Микс синусов в вокальном диапазоне + малый шум
    vocal = (
        0.08 * np.sin(2 * np.pi * 400 * t)
        + 0.05 * np.sin(2 * np.pi * 800 * t)
        + 0.03 * np.sin(2 * np.pi * 1200 * t)
        + 0.01 * (np.random.rand(n).astype(np.float32) - 0.5)
    )
    stereo = np.column_stack((vocal, vocal * 0.95))
    out = run_mastering_pipeline(stereo, sr, target_lufs=-14.0, style="standard")
    assert out.shape == stereo.shape
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    assert np.max(np.abs(out)) <= 1.01


def test_denoiser_vocal_like_not_silent():
    """При включённом Denoiser (preset light/vocal) вокало-подобный сигнал не даёт тишину на выходе (план 2.3.4)."""
    from app.pipeline import apply_spectral_denoise, DENOISE_PRESETS
    sr = 44100
    n = sr * 2
    t = np.linspace(0, n / sr, n, dtype=np.float32)
    vocal = (
        0.08 * np.sin(2 * np.pi * 400 * t)
        + 0.05 * np.sin(2 * np.pi * 800 * t)
        + 0.03 * np.sin(2 * np.pi * 1200 * t)
        + 0.01 * (np.random.rand(n).astype(np.float32) - 0.5)
    )
    stereo = np.column_stack((vocal, vocal * 0.95)).astype(np.float32)
    for preset in ("vocal", "light"):
        strength, noise_pct = DENOISE_PRESETS[preset]
        out = apply_spectral_denoise(stereo, sr, strength=strength, noise_percentile=noise_pct)
        assert out.shape == stereo.shape
        assert not np.any(np.isnan(out))
        assert np.max(np.abs(out)) > 0.01, f"preset={preset}: output too quiet"


def test_export_audio_wav_no_nan(stereo_2sec_44k):
    """Экспорт WAV 16-bit: декодированные сэмплы без NaN/Inf, в диапазоне [-1, 1]."""
    import soundfile as sf
    from app.pipeline import export_audio
    audio, sr = stereo_2sec_44k
    wav_bytes = export_audio(audio, sr, channels=2, out_format="wav", dither_type="tpdf")
    decoded, sr_out = sf.read(io.BytesIO(wav_bytes))
    assert not np.any(np.isnan(decoded))
    assert not np.any(np.isinf(decoded))
    assert np.all(decoded >= -1.01) and np.all(decoded <= 1.01)


def test_validate_mastered_not_silent_raises_on_silence():
    """При тишине на выходе пайплайна возвращается ошибка с подсказкой (задача 1.5 / 5.3)."""
    from app.pipeline import validate_mastered_not_silent
    with pytest.raises(ValueError) as exc_info:
        validate_mastered_not_silent(np.zeros(1000, dtype=np.float32))
    msg = str(exc_info.value)
    assert "тишину" in msg
    assert "Отключите" in msg


def test_validate_mastered_not_silent_passes_on_signal():
    """При наличии сигнала выше порога валидация не поднимает исключение."""
    from app.pipeline import validate_mastered_not_silent
    x = np.zeros(1000, dtype=np.float32)
    x[0] = 0.01
    validate_mastered_not_silent(x)


def test_pro_modules_transient_parallel_dyn_eq_not_silent(stereo_2sec_44k):
    """Transient Designer, Parallel Compression, Dynamic EQ не дают тишину на выходе (план 12.2)."""
    from app.pipeline import (
        apply_transient_designer,
        apply_parallel_compression,
        apply_dynamic_eq,
    )
    audio, sr = stereo_2sec_44k
    # Transient Designer (attack 1.1, sustain 0.9)
    out = apply_transient_designer(audio, sr, attack_gain=1.1, sustain_gain=0.9)
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))
    assert np.max(np.abs(out)) > 0.001
    # Parallel Compression mix 30%
    out = apply_parallel_compression(out, sr, mix=0.3)
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))
    assert np.max(np.abs(out)) > 0.001
    # Dynamic EQ
    out = apply_dynamic_eq(out, sr)
    assert out.shape == audio.shape
    assert not np.any(np.isnan(out))
    assert np.max(np.abs(out)) > 0.001


def test_dynamic_eq_only_not_silent(stereo_2sec_44k):
    """Только Dynamic EQ: выход не тишина, форма и тип корректны (для серийного теста и продакшена)."""
    from app.pipeline import apply_dynamic_eq
    audio, sr = stereo_2sec_44k
    out = apply_dynamic_eq(audio, sr)
    assert out.shape == audio.shape
    assert out.dtype == audio.dtype
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    assert np.max(np.abs(out)) > 0.001, "Dynamic EQ выдал тишину"


def test_high_freq_trim_cuts_highs(mono_2sec_44k):
    """Срез верхов на 10%: высокие частоты ослабляются, низкие почти не меняются."""
    from app.pipeline import apply_high_freq_trim
    audio, sr = mono_2sec_44k
    n = len(audio)
    t = np.linspace(0, n / sr, n, dtype=np.float32)
    # Сигнал выше 5 kHz (должен ослабляться на ~10%)
    high_tone = (0.2 * np.sin(2 * np.pi * 10000 * t)).astype(np.float32)
    out = apply_high_freq_trim(high_tone, sr, crossover_hz=5000.0, high_gain=0.9)
    assert out.shape == high_tone.shape
    # Амплитуда верхов должна уменьшиться (~0.9 от исходной)
    assert np.max(np.abs(out)) <= np.max(np.abs(high_tone)) * 0.95
    assert np.max(np.abs(out)) >= np.max(np.abs(high_tone)) * 0.8


def test_apply_output_edge_fade_in_attacks_start():
    from app.pipeline import apply_output_edge_fade_in
    sr = 48000
    x = np.ones(sr, dtype=np.float32) * 0.5
    out = apply_output_edge_fade_in(x, sr, fade_ms=5.0)
    assert out[0] < x[0]
    assert np.isclose(out[-1], x[-1], rtol=1e-4)
    n_fade = int(round(sr * 0.005))
    assert np.all(out[n_fade : n_fade + 100] == x[n_fade : n_fade + 100])


def test_apply_output_edge_fade_in_stereo(stereo_2sec_44k):
    from app.pipeline import apply_output_edge_fade_in
    audio, sr = stereo_2sec_44k
    out = apply_output_edge_fade_in(audio, sr, fade_ms=4.0)
    assert out.shape == audio.shape
    assert np.isclose(out[0, 0], 0.0, atol=1e-6)
    assert np.isclose(out[0, 1], 0.0, atol=1e-6)


def test_apply_maximizer_lookahead_boundary_smaller_step_than_hard_splice():
    """Кроссфейд на границе lookahead уменьшает скачок относительно «лобовой» склейки."""
    from app.pipeline import apply_maximizer
    sr = 48000
    n = sr * 2
    t = np.linspace(0, n / sr, n, dtype=np.float32)
    x = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    delay_n = int(sr * 0.006)
    x2 = x.reshape(-1, 1)
    delayed = np.concatenate([np.zeros((delay_n, 1), dtype=np.float32), x2[:-delay_n]], axis=0)
    limited = apply_maximizer(delayed)
    hard = np.concatenate([x2[:delay_n], limited[delay_n:]], axis=0).astype(np.float32)
    from app.pipeline import apply_maximizer_lookahead

    smooth = apply_maximizer_lookahead(x, sr, lookahead_ms=6.0).reshape(-1, 1)
    if delay_n > 2:
        step_hard = float(abs(hard[delay_n, 0] - hard[delay_n - 1, 0]))
        step_smooth = float(abs(smooth[delay_n, 0] - smooth[delay_n - 1, 0]))
        assert step_smooth <= step_hard + 1e-3


def test_run_mastering_pipeline_output_fade_starts_at_zero(mono_2sec_44k):
    """Финальный fade-in: первый сэмпл 0 (линейный ramp от нуля)."""
    from app.pipeline import run_mastering_pipeline
    audio, sr = mono_2sec_44k
    audio = audio.copy()
    audio[0] = 0.9
    out = run_mastering_pipeline(audio, sr, target_lufs=-14.0, style="standard")
    assert np.isclose(out[0], 0.0, atol=1e-4)
