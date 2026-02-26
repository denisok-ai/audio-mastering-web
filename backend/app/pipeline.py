# @file pipeline.py
# @description Конвейер мастеринга: DC, пики, EQ, компрессор, лимитер, нормализация LUFS
# @dependencies numpy, scipy, soundfile, pyloudnorm, pedalboard, librosa
# @created 2026-02-26

import io
from typing import Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pydub import AudioSegment

from .config import settings


# Целевые LUFS по пресетам (Sony/Warner/стриминг: -14 LUFS, True Peak ≤ -1 dBTP)
PRESET_LUFS = {
    "spotify": -14.0,
    "youtube": -14.0,
    "apple": -16.0,
    "club": -9.0,
    "broadcast": -24.0,
}

# Запас под межвыборочные пики (True Peak)
TRUE_PEAK_LIMIT_DB = -1.5

# Многополосная динамика по скриншотам: кроссоверы 214 Hz, 2.23 kHz, 10 kHz
MULTIBAND_CROSSOVERS_HZ = (214.0, 2230.0, 10000.0)
# По полосам: (limiter_thresh_db, comp_ratio, comp_thresh_db, gain_linear)
# Band 1: limiter -7.2, comp 1:1 off, gain 1.5
# Band 2: limiter -18.5, comp 2.2:1 -18.5, gain 1.5
# Band 3: limiter -26, comp 2.4:1 -26, gain 3.2
# Band 4: limiter -26.9, comp 3.7:1 -26.9, gain 3.6
MULTIBAND_CONFIG = (
    (-7.2, 1.0, -7.2, 1.5),
    (-18.5, 2.2, -18.5, 1.5),
    (-26.0, 2.4, -26.0, 3.2),
    (-26.9, 3.7, -26.9, 3.6),
)
# Максимайзер (скриншот 5): порог -2.5 dB, margin (потолок) -0.3 dB
MAXIMIZER_THRESHOLD_DB = -2.5
MAXIMIZER_MARGIN_DB = -0.3
# Финальный мастеринг по частотам (soothe2): trim +0.5 dB
FINAL_TRIM_DB = 0.5


def _pydub_to_numpy(seg: AudioSegment) -> Tuple[np.ndarray, int]:
    """Конвертация pydub AudioSegment в (samples, sample_rate)."""
    samples = np.array(seg.get_array_of_samples())
    if seg.channels == 2:
        samples = samples.reshape((-1, 2))
    sr = seg.frame_rate
    return samples.astype(np.float32) / (2**15), sr


def _numpy_to_pydub(samples: np.ndarray, sr: int, channels: int = 2) -> AudioSegment:
    """Конвертация numpy в pydub через WAV (soundfile) — надёжная запись без потери звука."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    samples = np.clip(samples, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return AudioSegment.from_wav(buf)


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Удаление DC-смещения по каждому каналу."""
    if audio.ndim == 1:
        return audio - np.mean(audio)
    return audio - np.mean(audio, axis=0, keepdims=True)


def remove_intersample_peaks(audio: np.ndarray, headroom_db: float = 0.5) -> np.ndarray:
    """Безопасное ограничение пиков с запасом (уменьшает межвыборочные пики)."""
    peak = np.nanmax(np.abs(audio))
    if not np.isfinite(peak) or peak <= 1e-12:
        return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    limit = 10 ** (-headroom_db / 20)
    if peak > limit:
        audio = audio * (limit / peak)
    return np.clip(audio, -1.0, 1.0)


def apply_target_curve(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Студийная кривая уровня Sony/Warner: обрезка румбла/ультразвука + лёгкий тональный баланс.
    - Highpass 40 Hz, Lowpass 18 kHz (чистота)
    - Лёгкий подъём «присутствия» 2–4 kHz (+0.35 dB), лёгкое снятие «грязи» 200–400 Hz (-0.25 dB).
    """
    from scipy import signal
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    nyq = sr / 2.0
    # Обрезка краёв
    f_hp = min(40.0 / nyq, 0.99)
    b_hp, a_hp = signal.butter(2, f_hp, btype="high", output="ba")
    f_lp = min(18000.0 / nyq, 0.99)
    b_lp, a_lp = signal.butter(2, f_lp, btype="low", output="ba")
    # Лёгкий «присутствие» (peak ~3 kHz, Q низкий), «снятие грязи» (peak ~300 Hz)
    g_presence = 10 ** (0.35 / 20)
    g_mud = 10 ** (-0.25 / 20)
    f_pres = min(3000.0 / nyq, 0.99)
    b_pres, a_pres = signal.butter(1, [f_pres * 0.7, f_pres * 1.3], btype="band", output="ba")
    f_mud = min(300.0 / nyq, 0.99)
    b_mud, a_mud = signal.butter(1, [f_mud * 0.7, f_mud * 1.3], btype="band", output="ba")
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        ch_out = signal.filtfilt(b_hp, a_hp, audio[:, ch])
        ch_out = signal.filtfilt(b_lp, a_lp, ch_out)
        pres = signal.filtfilt(b_pres, a_pres, ch_out)
        mud = signal.filtfilt(b_mud, a_mud, ch_out)
        ch_out = ch_out + (g_presence - 1.0) * pres + (g_mud - 1.0) * mud
        out[:, ch] = ch_out
    if audio.shape[1] == 1:
        return out[:, 0]
    return out


def _apply_limiter_numpy(audio: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
    """Лимитер по порогу (dB)."""
    limit = 10 ** (threshold_db / 20.0)
    return np.clip(audio, -limit, limit).astype(np.float32)


def _compress_soft_knee(audio: np.ndarray, threshold_db: float = -18.0, ratio: float = 2.5) -> np.ndarray:
    """Сжатие динамики (soft knee). Ratio 1.0 = bypass."""
    if ratio <= 1.0:
        return audio
    thresh = 10 ** (threshold_db / 20.0)
    abs_x = np.abs(audio)
    sign = np.sign(audio)
    excess = np.maximum(abs_x - thresh, 0.0)
    out_abs = np.minimum(abs_x, thresh + excess / ratio)
    return (sign * out_abs).astype(np.float32)


def _split_bands(audio: np.ndarray, sr: float, crossovers_hz: tuple) -> list:
    """Разделение на 4 полосы по кроссоверам (214, 2230, 10000 Hz). Butterworth 2nd order."""
    from scipy import signal
    nyq = sr / 2.0
    f1, f2, f3 = crossovers_hz[0] / nyq, crossovers_hz[1] / nyq, crossovers_hz[2] / nyq
    f1, f2, f3 = min(f1, 0.99), min(f2, 0.99), min(f3, 0.99)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    n_ch = audio.shape[1]
    bands = []
    for ch in range(n_ch):
        x = audio[:, ch]
        b_lp1, a_lp1 = signal.butter(2, f1, btype="low", output="ba")
        band1 = signal.filtfilt(b_lp1, a_lp1, x)
        b_hp2, a_hp2 = signal.butter(2, f1, btype="high", output="ba")
        b_lp2, a_lp2 = signal.butter(2, f2, btype="low", output="ba")
        band2 = signal.filtfilt(b_lp2, a_lp2, signal.filtfilt(b_hp2, a_hp2, x))
        b_hp3, a_hp3 = signal.butter(2, f2, btype="high", output="ba")
        b_lp3, a_lp3 = signal.butter(2, f3, btype="low", output="ba")
        band3 = signal.filtfilt(b_lp3, a_lp3, signal.filtfilt(b_hp3, a_hp3, x))
        b_hp4, a_hp4 = signal.butter(2, f3, btype="high", output="ba")
        band4 = signal.filtfilt(b_hp4, a_hp4, x)
        if ch == 0:
            bands = [band1.copy(), band2.copy(), band3.copy(), band4.copy()]
        else:
            bands[0] = np.column_stack([bands[0], band1])
            bands[1] = np.column_stack([bands[1], band2])
            bands[2] = np.column_stack([bands[2], band3])
            bands[3] = np.column_stack([bands[3], band4])
    if n_ch == 1:
        bands = [b.ravel() for b in bands]
    return bands


def _merge_bands(bands: list, n_channels: int) -> np.ndarray:
    """Сумма полос обратно в (samples, channels)."""
    out = bands[0] + bands[1] + bands[2] + bands[3]
    return out.astype(np.float32)


def apply_multiband_dynamics(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Многополосная динамика по скриншотам: 4 полосы (214 Hz, 2.23 kHz, 10 kHz).
    В каждой полосе: компрессор (ratio по скриншоту), лимитер 10:1, усиление полосы.
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n_ch = samples.shape[1]
    bands = _split_bands(samples, float(sr), MULTIBAND_CROSSOVERS_HZ)
    for i in range(4):
        lim_db, comp_ratio, comp_db, gain = MULTIBAND_CONFIG[i]
        bands[i] = _compress_soft_knee(bands[i], threshold_db=comp_db, ratio=comp_ratio)
        bands[i] = _apply_limiter_numpy(bands[i], threshold_db=lim_db)
        bands[i] = bands[i] * gain
    out = _merge_bands(bands, n_ch)
    if n_ch == 1 and out.ndim == 1:
        return out
    if n_ch == 1:
        return out[:, 0]
    return out


def apply_maximizer(audio: np.ndarray) -> np.ndarray:
    """Максимайзер по скриншоту 5: порог -2.5 dB, потолок (margin) -0.3 dB."""
    ceiling = 10 ** (MAXIMIZER_MARGIN_DB / 20.0)
    thresh = 10 ** (MAXIMIZER_THRESHOLD_DB / 20.0)
    abs_x = np.abs(audio)
    sign = np.sign(audio)
    out_abs = np.where(abs_x <= thresh, abs_x, thresh + (abs_x - thresh) * (ceiling - thresh) / (1.0 - thresh))
    out_abs = np.minimum(out_abs, ceiling)
    return (sign * out_abs).astype(np.float32)


def apply_final_spectral_balance(audio: np.ndarray, sr: int) -> np.ndarray:
    """Финальный мастеринг по частотам (по скриншоту soothe2): лёгкие коррекции + trim +0.5 dB."""
    from scipy import signal
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    nyq = sr / 2.0
    out = audio.copy()
    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        dip3k = 10 ** (-0.5 / 20)
        dip16k = 10 ** (-0.3 / 20)
        lift_low = 10 ** (0.3 / 20)
        lift8k = 10 ** (0.2 / 20)
        f_3k = min(3000.0 / nyq, 0.99)
        b_3k, a_3k = signal.butter(1, [f_3k * 0.8, f_3k * 1.2], btype="band", output="ba")
        band_3k = signal.filtfilt(b_3k, a_3k, x)
        f_16k = min(16000.0 / nyq, 0.99)
        b_16k, a_16k = signal.butter(2, f_16k, btype="high", output="ba")
        band_16k = signal.filtfilt(b_16k, a_16k, x)
        f_low = min(180.0 / nyq, 0.99)
        b_low, a_low = signal.butter(2, f_low, btype="low", output="ba")
        band_low = signal.filtfilt(b_low, a_low, x)
        f_8k = min(8000.0 / nyq, 0.99)
        b_8k, a_8k = signal.butter(1, [f_8k * 0.8, f_8k * 1.2], btype="band", output="ba")
        band_8k = signal.filtfilt(b_8k, a_8k, x)
        x = x + (dip3k - 1.0) * band_3k * 0.25 + (dip16k - 1.0) * band_16k * 0.25
        x = x + (lift_low - 1.0) * band_low * 0.25 + (lift8k - 1.0) * band_8k * 0.25
        trim = 10 ** (FINAL_TRIM_DB / 20.0)
        out[:, ch] = x * trim
    if audio.shape[1] == 1:
        return out[:, 0]
    return out


def apply_dynamics(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Динамика по скриншотам: многополосная обработка (4 полосы) + максимайзер.
    Сохраняется форма волны, контролируются только пики и баланс полос.
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples = apply_multiband_dynamics(samples, sr)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    samples = apply_maximizer(samples)
    samples = _apply_limiter_numpy(samples, threshold_db=TRUE_PEAK_LIMIT_DB)
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return samples[:, 0]
    return samples


def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Нормализация к целевому LUFS. Ограничение усиления ±20 dB — защита от артефактов при сломанном сигнале."""
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return audio
    delta_db = target_lufs - loudness
    max_gain_db = 20.0
    delta_db = np.clip(delta_db, -max_gain_db, max_gain_db)
    gain = 10 ** (delta_db / 20.0)
    return (audio * gain).astype(np.float32)


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Измерение текущей громкости в LUFS."""
    meter = pyln.Meter(sr)
    try:
        return float(meter.integrated_loudness(audio))
    except Exception:
        return float("nan")


def load_audio_from_bytes(data: bytes, fmt: str) -> Tuple[np.ndarray, int]:
    """Загрузка аудио из байтов (WAV/MP3/FLAC) через pydub. Возвращает (samples, sr)."""
    # fmt может быть полным именем файла (Untitled.wav) — берём только расширение
    if "." in fmt:
        ext = fmt.rsplit(".", 1)[-1].lower()
    else:
        ext = fmt.lower().lstrip(".")
    if ext == "wav":
        seg = AudioSegment.from_wav(io.BytesIO(data))
    elif ext == "mp3":
        seg = AudioSegment.from_mp3(io.BytesIO(data))
    elif ext == "flac":
        seg = AudioSegment.from_file(io.BytesIO(data), format="flac")
    else:
        seg = AudioSegment.from_file(io.BytesIO(data), format=ext)
    return _pydub_to_numpy(seg)


def _write_wav_16bit_dithered(buf: io.BytesIO, samples: np.ndarray, sr: int) -> None:
    """WAV 16-bit с TPDF-дизерингом (практика крупных лейблов)."""
    scale = 32767.0
    tpdf = (np.random.rand(*samples.shape) + np.random.rand(*samples.shape) - 1.0).astype(np.float32)
    dithered = samples * scale + tpdf
    int16 = np.clip(np.round(dithered), -32768, 32767).astype(np.int16)
    sf.write(buf, int16, sr, format="WAV", subtype="PCM_16")


def export_audio(
    samples: np.ndarray,
    sr: int,
    channels: int,
    out_format: str = "wav",
) -> bytes:
    """Экспорт в wav/mp3/flac. WAV: 16-bit с TPDF-дизерингом."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    samples = np.clip(samples, -1.0, 1.0)

    if out_format == "wav":
        buf = io.BytesIO()
        _write_wav_16bit_dithered(buf, samples, sr)
        buf.seek(0)
        return buf.getvalue()

    wav_buf = io.BytesIO()
    _write_wav_16bit_dithered(wav_buf, samples, sr)
    wav_buf.seek(0)
    seg = AudioSegment.from_wav(wav_buf)
    out_buf = io.BytesIO()
    if out_format == "mp3":
        seg.export(out_buf, format="mp3", bitrate="320k")
    elif out_format == "flac":
        seg.export(out_buf, format="flac")
    else:
        seg.export(out_buf, format=out_format)
    return out_buf.getvalue()


def run_mastering_pipeline(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    progress_callback=None,
) -> np.ndarray:
    """
    Конвейер студийного мастеринга по скриншотам.
    progress_callback(percent: int, message: str) вызывается на каждом шаге (0–100).
    """
    def report(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    report(5, "Подготовка…")
    audio = remove_dc_offset(audio)
    report(12, "Удаление DC-смещения")
    audio = remove_intersample_peaks(audio, headroom_db=0.5)
    report(20, "Защита от пиков")
    audio = apply_target_curve(audio, sr)
    report(35, "Студийный EQ")
    audio = apply_dynamics(audio, sr)
    report(55, "Многополосная динамика и максимайзер")
    audio = normalize_lufs(audio, sr, target_lufs)
    report(70, "Нормализация LUFS")
    audio = apply_final_spectral_balance(audio, sr)
    report(82, "Финальная частотная коррекция")
    audio = remove_intersample_peaks(audio, headroom_db=0.5)
    report(90, "Финальная защита пиков")
    out = np.clip(audio, -1.0, 1.0).astype(np.float32)
    out = np.ascontiguousarray(out)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    report(95, "Готово")
    return out
