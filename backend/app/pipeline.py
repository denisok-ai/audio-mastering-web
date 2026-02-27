# @file pipeline.py
# @description Конвейер мастеринга: DC, пики, EQ, компрессор, лимитер, нормализация LUFS
# @dependencies numpy, scipy, soundfile, pyloudnorm, pydub
# @created 2026-02-26

import io
from typing import NoReturn, Tuple

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

# Жанровые пресеты: целевой LUFS + мягкие EQ-корректировки по частотным полосам.
# Поля: lufs, sub (≤80Hz), bass (80–250Hz), mids (800–2500Hz),
#        presence (3–8kHz), air (10–18kHz), comp_mult (множитель компрессии),
#        exciter_db (гармонический эксайтер, дБ, аналог iZotope Ozone 5 Exciter),
#        imager_width (стерео-расширение 0.0=моно, 1.0=оригинал, 1.5=wide, аналог Imager)
STYLE_CONFIGS: dict[str, dict] = {
    "standard":    {"lufs": -14.0, "sub":  0.0, "bass":  0.0, "mids":  0.0, "presence":  0.0, "air":  0.0, "comp_mult": 1.0,  "exciter_db": 0.0, "imager_width": 1.0},
    "edm":         {"lufs":  -9.0, "sub":  1.8, "bass":  0.9, "mids": -0.3, "presence":  0.6, "air":  0.9, "comp_mult": 1.3,  "exciter_db": 0.6, "imager_width": 1.25},
    "hiphop":      {"lufs": -13.0, "sub":  1.4, "bass":  0.7, "mids":  0.5, "presence":  0.3, "air":  0.2, "comp_mult": 1.2,  "exciter_db": 0.3, "imager_width": 1.1},
    "classical":   {"lufs": -18.0, "sub": -0.5, "bass":  0.0, "mids":  0.0, "presence":  0.3, "air":  0.6, "comp_mult": 0.45, "exciter_db": 0.0, "imager_width": 1.05},
    "podcast":     {"lufs": -16.0, "sub": -1.2, "bass": -0.4, "mids":  0.9, "presence":  0.7, "air":  0.0, "comp_mult": 1.1,  "exciter_db": 0.0, "imager_width": 1.0},
    "lofi":        {"lufs": -18.0, "sub":  0.4, "bass":  0.6, "mids": -0.6, "presence": -1.0, "air": -1.8, "comp_mult": 0.65, "exciter_db": 0.2, "imager_width": 0.9},
    # House Basic — пресет на основе анализа iZotope Ozone 5:
    # Equalizer: буст суб/басов + воздух; срез мутных мидов
    # Dynamics: tight 4-band compression (comp_mult 1.35)
    # Maximizer: клубная громкость −10 LUFS
    # Exciter: лёгкое гармоническое насыщение верхних частот (Ozone 5 Exciter)
    # Imager: широкая стереобаза (Ozone 5 Imager)
    "house_basic": {"lufs": -10.0, "sub":  1.8, "bass":  0.9, "mids": -0.5, "presence":  0.8, "air":  1.0, "comp_mult": 1.35, "exciter_db": 0.8, "imager_width": 1.3},
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


def _target_curve_iir_coeffs(sr: int):
    """Коэффициенты IIR для студийной кривой (HP 40, LP 18k, presence 3k, mud 300 Hz). Возвращает (b_hp, a_hp, b_lp, a_lp, b_pres, a_pres, b_mud, a_mud), g_presence, g_mud."""
    from scipy import signal
    nyq = sr / 2.0
    f_hp = min(40.0 / nyq, 0.99)
    b_hp, a_hp = signal.butter(2, f_hp, btype="high", output="ba")
    f_lp = min(18000.0 / nyq, 0.99)
    b_lp, a_lp = signal.butter(2, f_lp, btype="low", output="ba")
    g_presence = 10 ** (0.35 / 20)
    g_mud = 10 ** (-0.25 / 20)
    f_pres = min(3000.0 / nyq, 0.99)
    b_pres, a_pres = signal.butter(1, [f_pres * 0.7, f_pres * 1.3], btype="band", output="ba")
    f_mud = min(300.0 / nyq, 0.99)
    b_mud, a_mud = signal.butter(1, [f_mud * 0.7, f_mud * 1.3], btype="band", output="ba")
    return (b_hp, a_hp, b_lp, a_lp, b_pres, a_pres, b_mud, a_mud), g_presence, g_mud


def _build_linear_phase_ir(sr: int, n_fft: int = 4096) -> np.ndarray:
    """
    Строит импульсную характеристику линейно-фазового EQ по той же кривой, что и apply_target_curve.
    IR имеет длину n_fft, задержка (n_fft-1)/2 сэмплов. Возвращает 1D array float32.
    """
    from scipy import signal
    coeffs, g_presence, g_mud = _target_curve_iir_coeffs(sr)
    b_hp, a_hp, b_lp, a_lp, b_pres, a_pres, b_mud, a_mud = coeffs
    # Частотная характеристика на сетке 0..n_fft//2
    w = np.pi * np.arange(n_fft // 2 + 1) / (n_fft // 2) if n_fft > 0 else np.array([0.0])
    _, h_hp = signal.freqz(b_hp, a_hp, worN=w)
    _, h_lp = signal.freqz(b_lp, a_lp, worN=w)
    _, h_pres = signal.freqz(b_pres, a_pres, worN=w)
    _, h_mud = signal.freqz(b_mud, a_mud, worN=w)
    # Суммарная АЧХ: HP*LP*(1 + (g_pres-1)*H_pres + (g_mud-1)*H_mud)
    H = h_hp * h_lp * (1.0 + (g_presence - 1.0) * h_pres + (g_mud - 1.0) * h_mud)
    H_mag = np.abs(H)
    H_mag = np.clip(H_mag, 1e-8, 1e8)
    # Линейная фаза: задержка (n_fft-1)/2 сэмплов → H(k) = H_mag * exp(-j*2*pi*k*(N-1)/(2*N))
    N = n_fft
    k_pos = np.arange(N // 2 + 1, dtype=np.float64)
    phase = -2.0 * np.pi * k_pos * (N - 1) / (2.0 * N)
    H_full = np.zeros(N, dtype=np.complex128)
    H_full[: N // 2 + 1] = H_mag * np.exp(1j * phase)
    for k in range(1, N // 2):
        H_full[N - k] = np.conj(H_full[k])
    if N % 2 == 0:
        H_full[N // 2] = np.real(H_full[N // 2])
    ir = np.fft.ifft(H_full).real
    ir = np.ascontiguousarray(ir.astype(np.float32))
    return ir


def apply_target_curve_linear_phase(audio: np.ndarray, sr: int, n_fft: int = 4096) -> np.ndarray:
    """
    Студийная кривая (Sony/Warner) в линейно-фазовом режиме: FFT overlap-add, задержка (n_fft-1)/2 сэмплов.
    Та же АЧХ, что и apply_target_curve; фаза линейная (без фазовых искажений).
    """
    from scipy import signal
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    ir = _build_linear_phase_ir(sr, n_fft)
    out = np.zeros_like(audio, dtype=np.float32)
    for ch in range(audio.shape[1]):
        out[:, ch] = signal.fftconvolve(audio[:, ch], ir, mode="same")
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    if audio.shape[1] == 1:
        return out[:, 0]
    return out


def apply_target_curve(
    audio: np.ndarray, sr: int, phase_mode: str = "minimum", eq_ms: bool = False
) -> np.ndarray:
    """
    Студийная кривая уровня Sony/Warner: обрезка румбла/ультразвука + лёгкий тональный баланс.
    - Highpass 40 Hz, Lowpass 18 kHz (чистота)
    - Лёгкий подъём «присутствия» 2–4 kHz (+0.35 dB), лёгкое снятие «грязи» 200–400 Hz (-0.25 dB).
    phase_mode: "minimum" — IIR filtfilt; "linear_phase" — FFT overlap-add.
    eq_ms: при стерео — применять кривую отдельно к Mid и Side (M/S per-band), затем собрать L/R.
    """
    if audio.ndim == 2 and audio.shape[1] == 2 and eq_ms:
        mid = ((audio[:, 0] + audio[:, 1]) * 0.5).astype(np.float32)
        side = ((audio[:, 0] - audio[:, 1]) * 0.5).astype(np.float32)
        mid_out = apply_target_curve(mid, sr, phase_mode=phase_mode, eq_ms=False)
        side_out = apply_target_curve(side, sr, phase_mode=phase_mode, eq_ms=False)
        out_l = np.clip(mid_out + side_out, -1.0, 1.0).astype(np.float32)
        out_r = np.clip(mid_out - side_out, -1.0, 1.0).astype(np.float32)
        return np.stack([out_l, out_r], axis=1)
    if phase_mode == "linear_phase":
        return apply_target_curve_linear_phase(audio, sr)
    from scipy import signal
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    coeffs, g_presence, g_mud = _target_curve_iir_coeffs(sr)
    b_hp, a_hp, b_lp, a_lp, b_pres, a_pres, b_mud, a_mud = coeffs
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


def _compress_soft_knee(
    audio: np.ndarray,
    threshold_db: float = -18.0,
    ratio: float = 2.5,
    knee_db: float = 6.0,
    max_upward_boost_db: float = 12.0,
) -> np.ndarray:
    """
    Сжатие/расширение динамики. ratio > 1 — downward (компрессия), ratio < 1 — upward (усиление тихих).
    knee_db: мягкое колено для downward. max_upward_boost_db: макс. подъём при upward (защита от шума).
    Ratio 1.0 = bypass.
    """
    if ratio <= 0.0:
        return audio
    thresh_lin = 10 ** (threshold_db / 20.0)
    abs_x = np.abs(audio)
    sign = np.sign(audio)
    eps = 1e-12

    if ratio < 1.0:
        level_db = np.where(abs_x > eps, 20.0 * np.log10(np.maximum(abs_x, eps)), -100.0)
        boost_db = (threshold_db - level_db) * (1.0 - ratio)
        boost_db = np.clip(boost_db, 0.0, max(0.1, float(max_upward_boost_db)))
        gain_lin = 10 ** (boost_db / 20.0)
        out_abs = np.clip(abs_x * gain_lin, 0.0, 1.0)
        return (sign * out_abs).astype(np.float32)

    if ratio == 1.0:
        return audio
    knee_db = max(0.0, float(knee_db))
    if knee_db < 0.5:
        excess = np.maximum(abs_x - thresh_lin, 0.0)
        out_abs = np.minimum(abs_x, thresh_lin + excess / ratio)
        return (sign * out_abs).astype(np.float32)
    lower = thresh_lin * 10 ** (-knee_db / 20.0)
    upper = thresh_lin * 10 ** (knee_db / 20.0)
    out_abs = np.where(
        abs_x <= lower,
        abs_x,
        np.where(
            abs_x >= upper,
            thresh_lin + (abs_x - thresh_lin) / ratio,
            lower + (abs_x - lower) * (
                (thresh_lin + (upper - thresh_lin) / ratio - lower) / (upper - lower)
            ),
        ),
    )
    out_abs = np.clip(out_abs, 0.0, None)
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


def apply_multiband_dynamics(
    samples: np.ndarray,
    sr: int,
    knee_db: float = 6.0,
    crossovers_hz: tuple[float, float, float] | None = None,
    band_ratios: tuple[float, float, float, float] | list[float] | None = None,
    max_upward_boost_db: float = 12.0,
) -> np.ndarray:
    """
    Многополосная динамика: 4 полосы по кроссоверам.
    band_ratios: опционально (r0,r1,r2,r3) — ratio по полосам; ratio < 1 = upward (усиление тихих).
    max_upward_boost_db: макс. подъём при upward (защита от шума).
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    n_ch = samples.shape[1]
    cross = crossovers_hz if crossovers_hz and len(crossovers_hz) == 3 else MULTIBAND_CROSSOVERS_HZ
    cross = tuple(float(np.clip(c, 20.0, 20000.0)) for c in cross)
    if cross[0] >= cross[1] or cross[1] >= cross[2]:
        cross = MULTIBAND_CROSSOVERS_HZ
    bands = _split_bands(samples, float(sr), cross)
    ratios_override = None
    if band_ratios is not None and len(band_ratios) == 4:
        ratios_override = tuple(float(band_ratios[i]) for i in range(4))
    for i in range(4):
        lim_db, comp_ratio, comp_db, gain = MULTIBAND_CONFIG[i]
        ratio = ratios_override[i] if ratios_override else comp_ratio
        bands[i] = _compress_soft_knee(
            bands[i],
            threshold_db=comp_db,
            ratio=ratio,
            knee_db=knee_db,
            max_upward_boost_db=max_upward_boost_db,
        )
        bands[i] = _apply_limiter_numpy(bands[i], threshold_db=lim_db)
        bands[i] = bands[i] * gain
    out = _merge_bands(bands, n_ch)
    if n_ch == 1 and out.ndim == 1:
        return out
    if n_ch == 1:
        return out[:, 0]
    return out


def apply_maximizer(audio: np.ndarray) -> np.ndarray:
    """Максимайзер по скриншоту 5: порог -2.5 dB, потолок (margin) -0.3 dB (soft_knee)."""
    ceiling = 10 ** (MAXIMIZER_MARGIN_DB / 20.0)
    thresh = 10 ** (MAXIMIZER_THRESHOLD_DB / 20.0)
    abs_x = np.abs(audio)
    sign = np.sign(audio)
    out_abs = np.where(abs_x <= thresh, abs_x, thresh + (abs_x - thresh) * (ceiling - thresh) / (1.0 - thresh))
    out_abs = np.minimum(out_abs, ceiling)
    return (sign * out_abs).astype(np.float32)


def _envelope_follower(x: np.ndarray, sr: float, attack_sec: float, release_sec: float) -> np.ndarray:
    """Однополюсный envelope follower: attack_sec и release_sec в секундах."""
    n = len(x)
    if n == 0:
        return x
    attack_coef = np.exp(-1.0 / max(1e-6, sr * attack_sec))
    release_coef = np.exp(-1.0 / max(1e-6, sr * release_sec))
    env = np.empty(n, dtype=np.float32)
    env[0] = abs(x[0])
    for i in range(1, n):
        val = abs(x[i])
        if val > env[i - 1]:
            env[i] = attack_coef * env[i - 1] + (1.0 - attack_coef) * val
        else:
            env[i] = release_coef * env[i - 1] + (1.0 - release_coef) * val
    return env


def apply_maximizer_transient_aware(
    audio: np.ndarray,
    sr: int,
    sensitivity: float = 0.5,
) -> np.ndarray:
    """
    Максимайзер с сохранением атак (transient-aware): детектор onset + переменный gain.
    При обнаружении транзиента доля ограничения снижается (sensitivity 0–1).
    """
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    limited = apply_maximizer(audio)
    mono = np.mean(np.abs(audio), axis=1).astype(np.float32)
    fast = _envelope_follower(mono, float(sr), 0.0005, 0.002)
    slow = _envelope_follower(mono, float(sr), 0.01, 0.04)
    diff = np.maximum(fast - slow, 0.0)
    denom = slow + 1e-12
    mask = np.clip(diff / denom * float(sensitivity), 0.0, 1.0)
    mask = np.minimum(mask, 1.0)
    for ch in range(audio.shape[1]):
        limited[:, ch] = limited[:, ch] * (1.0 - mask) + audio[:, ch] * mask
    limited = np.clip(limited, -1.0, 1.0).astype(np.float32)
    if audio.shape[1] == 1:
        return limited[:, 0]
    return limited


def apply_maximizer_lookahead(audio: np.ndarray, sr: int, lookahead_ms: float = 6.0) -> np.ndarray:
    """
    Максимайзер с lookahead: буфер 6 ms, лимитер «видит» пик заранее.
    Первые lookahead_ms мс проходят без ограничения, остальное — через тот же soft_knee максимайзер.
    """
    delay_n = int(sr * (lookahead_ms / 1000.0))
    if delay_n <= 0 or delay_n >= audio.shape[0]:
        return apply_maximizer(audio)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    delayed = np.concatenate([
        np.zeros((delay_n, audio.shape[1]), dtype=audio.dtype),
        audio[:-delay_n],
    ], axis=0)
    limited = apply_maximizer(delayed)
    out = np.concatenate([audio[:delay_n], limited[delay_n:]], axis=0).astype(np.float32)
    if audio.shape[1] == 1:
        return out[:, 0]
    return out


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


def apply_dynamics(
    samples: np.ndarray,
    sr: int,
    knee_db: float = 6.0,
    crossovers_hz: tuple[float, float, float] | None = None,
    band_ratios: tuple[float, float, float, float] | list[float] | None = None,
    max_upward_boost_db: float = 12.0,
) -> np.ndarray:
    """
    Динамика: многополосная обработка (4 полосы) + максимайзер + лимитер.
    band_ratios: опционально [r0,r1,r2,r3]; ratio < 1 = upward по полосе.
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples = apply_multiband_dynamics(
        samples,
        sr,
        knee_db=knee_db,
        crossovers_hz=crossovers_hz,
        band_ratios=band_ratios,
        max_upward_boost_db=max_upward_boost_db,
    )
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


def compute_lufs_timeline(
    audio: np.ndarray,
    sr: int,
    block_sec: float = 0.4,
    max_points: int = 300,
) -> tuple[list[float | None], float]:
    """
    Краткосрочный LUFS по времени для графика громкости.
    Аудио разбивается на блоки по block_sec (400 ms), для каждого блока считается
    integrated_loudness. Возвращает (список LUFS по времени, шаг по времени в секундах).
    """
    duration_sec = len(audio) / sr
    block_samples = int(sr * block_sec)
    if duration_sec <= block_sec or audio.size < block_samples:
        lufs = measure_lufs(audio, sr)
        return ([round(lufs, 2)] if not np.isnan(lufs) else [None], 0.0)
    n_points = min(max_points, max(1, int((duration_sec - block_sec) / (block_sec * 0.25)) + 1))
    step_sec = (duration_sec - block_sec) / max(n_points - 1, 1)
    step_samples = int(sr * step_sec)
    meter = pyln.Meter(sr)
    result: list[float | None] = []
    pos = 0
    while pos + block_samples <= len(audio) and len(result) < max_points:
        segment = audio[pos : pos + block_samples]
        try:
            val = float(meter.integrated_loudness(segment))
            result.append(round(val, 2))
        except Exception:
            result.append(None)
        pos += step_samples
    return (result, round(step_sec, 4))


def compute_spectrum_bars(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    n_bars: int = 64,
    min_hz: float = 20.0,
    max_hz: float = 20000.0,
) -> list[float]:
    """
    Логарифмические полосы спектра (dB) для анализа. Берётся срез из середины, Hann окно, FFT.
    Возвращает n_bars значений в dB (примерно от -80 до 0).
    """
    if audio.size < n_fft:
        return [-80.0] * n_bars
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = np.asarray(audio, dtype=np.float64)
    n = len(mono)
    start = max(0, n // 2 - n_fft // 2)
    frame = mono[start : start + n_fft].copy()
    window = np.hanning(n_fft)
    frame *= window
    spec = np.fft.rfft(frame)
    mag = np.abs(spec) * (2.0 / n_fft)
    n_bins = len(mag)
    nyq = sr / 2.0
    bars = []
    for b in range(n_bars):
        f_log = min_hz * (max_hz / min_hz) ** (b / max(n_bars - 1, 1))
        f_next = min_hz * (max_hz / min_hz) ** ((b + 1) / max(n_bars - 1, 1))
        k0 = max(0, int((f_log / nyq) * (n_fft // 2)))
        k1 = min(n_bins - 1, int(np.ceil((f_next / nyq) * (n_fft // 2))))
        if k0 > k1:
            peak = 1e-12
        else:
            peak = float(np.max(mag[k0 : k1 + 1]))
        db = 20.0 * np.log10(max(peak, 1e-12))
        bars.append(round(db, 2))
    return bars


def compute_vectorscope_points(
    audio: np.ndarray,
    max_points: int = 1000,
) -> list[list[float]]:
    """
    Точки для векторскопа (Lissajous L vs R): массив [l, r] в диапазоне [-1, 1].
    Для стерео — равномерная выборка по времени; для моно возвращает [].
    """
    if audio.ndim != 2 or audio.shape[1] != 2 or audio.size < 4:
        return []
    left = np.asarray(audio[:, 0], dtype=np.float64)
    right = np.asarray(audio[:, 1], dtype=np.float64)
    n = len(left)
    step = max(1, n // max_points)
    points: list[list[float]] = []
    for i in range(0, n, step):
        l_val = float(np.clip(left[i], -1.0, 1.0))
        r_val = float(np.clip(right[i], -1.0, 1.0))
        points.append([round(l_val, 5), round(r_val, 5)])
        if len(points) >= max_points:
            break
    return points


def measure_stereo_correlation(audio: np.ndarray) -> float | None:
    """
    Корреляция между L и R каналами (correlation meter).
    Возвращает значение в [-1, 1]: -1 = противофаза, 0 = стерео, +1 = моно.
    Для моно или не-стерео возвращает None.
    """
    if audio.ndim != 2 or audio.shape[1] != 2 or audio.size < 4:
        return None
    left = np.asarray(audio[:, 0], dtype=np.float64)
    right = np.asarray(audio[:, 1], dtype=np.float64)
    n = left.size
    sum_l = np.sum(left)
    sum_r = np.sum(right)
    sum_lr = np.sum(left * right)
    sum_l2 = np.sum(left * left)
    sum_r2 = np.sum(right * right)
    denom = np.sqrt(max(sum_l2 * sum_r2, 0.0))
    if denom < 1e-20:
        return None
    # Pearson correlation
    num = n * sum_lr - sum_l * sum_r
    denom2 = np.sqrt(max(n * sum_l2 - sum_l * sum_l, 0.0)) * np.sqrt(max(n * sum_r2 - sum_r * sum_r, 0.0))
    if denom2 < 1e-20:
        return 0.0
    r = num / denom2
    return float(np.clip(r, -1.0, 1.0))


def _raise_ffmpeg_error(fmt: str, cause: Exception) -> NoReturn:
    """Выбрасывает понятную ошибку когда ffmpeg/ffprobe не установлен."""
    raise RuntimeError(
        f"Формат {fmt.upper()} требует ffmpeg, который не найден на сервере. "
        "Установите: sudo apt-get install -y ffmpeg"
    ) from cause


def load_audio_from_bytes(data: bytes, fmt: str) -> Tuple[np.ndarray, int]:
    """Загрузка аудио из байтов (WAV/MP3/FLAC). Возвращает (samples, sr).

    WAV и FLAC читаются через soundfile — ffmpeg НЕ требуется.
    MP3 и прочие форматы используют pydub и требуют ffmpeg/ffprobe.
    """
    if "." in fmt:
        ext = fmt.rsplit(".", 1)[-1].lower()
    else:
        ext = fmt.lower().lstrip(".")

    # WAV и FLAC: soundfile читает без ffmpeg
    if ext in ("wav", "flac"):
        buf = io.BytesIO(data)
        samples, sr = sf.read(buf, dtype="float32", always_2d=True)
        return samples, int(sr)

    # MP3 и прочие форматы: pydub (требует ffmpeg)
    try:
        if ext == "mp3":
            seg = AudioSegment.from_mp3(io.BytesIO(data))
        else:
            seg = AudioSegment.from_file(io.BytesIO(data), format=ext)
    except FileNotFoundError as exc:
        _raise_ffmpeg_error(ext, exc)
    return _pydub_to_numpy(seg)


def _dither_noise_tpdf(shape: tuple) -> np.ndarray:
    """TPDF (triangular PDF) dither: сумма двух uniform − 1, макс. амплитуда 1 LSB."""
    return (np.random.rand(*shape) + np.random.rand(*shape) - 1.0).astype(np.float32)


def _dither_noise_ns_e(shape: tuple) -> np.ndarray:
    """Упрощённый noise-shaped dither (E-weighted style): high-pass фильтр сдвигает шум вверх по частоте."""
    n = shape[0] if shape else 0
    ch = shape[1] if len(shape) > 1 else 1
    if n < 4:
        return _dither_noise_tpdf(shape)
    white = (np.random.rand(n, ch) if ch > 1 else np.random.rand(n)).astype(np.float32)
    white = 2.0 * white - 1.0
    # Простой 1-pole HP: y[n] = x[n] - x[n-1] + 0.99*y[n-1] → шум смещается в высокие частоты
    out = np.empty_like(white)
    if white.ndim == 1:
        out[0] = white[0]
        for i in range(1, n):
            out[i] = white[i] - white[i - 1] + 0.99 * out[i - 1]
    else:
        out[0, :] = white[0, :]
        for i in range(1, n):
            out[i, :] = white[i, :] - white[i - 1, :] + 0.99 * out[i - 1, :]
    return out.astype(np.float32)


def _dither_noise_ns_itu(shape: tuple) -> np.ndarray:
    """Noise-shaped dither (ITU-style): двухполюсный HP, сдвиг шума в высокие частоты (broadcast/ITU)."""
    n = shape[0] if shape else 0
    ch = shape[1] if len(shape) > 1 else 1
    if n < 8:
        return _dither_noise_tpdf(shape)
    white = (np.random.rand(n, ch) if ch > 1 else np.random.rand(n)).astype(np.float32)
    white = 2.0 * white - 1.0
    # 2-pole HP: больше сдвига в highs, меньше остаточного шума в низких
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([1.0, -1.96, 0.9604])
    out = np.empty_like(white)
    if white.ndim == 1:
        from scipy import signal as _sg
        out[:] = _sg.lfilter(b, a, white)
    else:
        from scipy import signal as _sg
        for c in range(white.shape[1]):
            out[:, c] = _sg.lfilter(b, a, white[:, c])
    return out.astype(np.float32)


def _write_wav_16bit_dithered(
    buf: io.BytesIO,
    samples: np.ndarray,
    sr: int,
    dither_type: str = "tpdf",
) -> None:
    """WAV 16-bit с дизерингом. dither_type: tpdf, ns_e (noise-shaped), ns_itu (ITU-style)."""
    scale = 32767.0
    if dither_type == "ns_e":
        noise = _dither_noise_ns_e(samples.shape)
    elif dither_type == "ns_itu":
        noise = _dither_noise_ns_itu(samples.shape)
    else:
        noise = _dither_noise_tpdf(samples.shape)
    dithered = samples * scale + noise
    int16 = np.clip(np.round(dithered), -32768, 32767).astype(np.int16)
    sf.write(buf, int16, sr, format="WAV", subtype="PCM_16")


def _auto_blank_end(samples: np.ndarray, sr: int, threshold_dbfs: float = -60.0, min_silence_sec: float = 0.5) -> np.ndarray:
    """Обрезает тишину в конце. threshold_dbfs: порог (dBFS), min_silence_sec: мин. длительность тишины для среза."""
    if samples.size == 0 or min_silence_sec <= 0:
        return samples
    threshold_lin = 10 ** (threshold_dbfs / 20.0)
    n_silence = int(sr * min_silence_sec)
    if n_silence <= 0:
        return samples
    n = samples.shape[0]
    peak = np.max(np.abs(samples), axis=1) if samples.ndim > 1 else np.abs(samples)
    idx = n
    for i in range(n - 1, -1, -1):
        if peak[i] > threshold_lin:
            idx = min(n, i + 1 + n_silence)
            break
    return samples[:idx]


def export_audio(
    samples: np.ndarray,
    sr: int,
    channels: int,
    out_format: str = "wav",
    dither_type: str = "tpdf",
    auto_blank_sec: float = 0.0,
) -> bytes:
    """Экспорт в wav/mp3/flac.

    WAV: 16-bit с дизерингом (dither_type: tpdf или ns_e). FLAC: 24-bit. MP3: 320 kbps (ffmpeg).
    auto_blank_sec > 0: обрезка тишины в конце перед экспортом.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    samples = np.clip(samples, -1.0, 1.0)
    if auto_blank_sec > 0:
        samples = _auto_blank_end(samples, sr, threshold_dbfs=-50.0, min_silence_sec=auto_blank_sec)

    if out_format == "wav":
        buf = io.BytesIO()
        _write_wav_16bit_dithered(buf, samples, sr, dither_type=dither_type)
        buf.seek(0)
        return buf.getvalue()

    if out_format == "flac":
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="FLAC", subtype="PCM_24")
        buf.seek(0)
        return buf.getvalue()

    if out_format == "mp3":
        try:
            wav_buf = io.BytesIO()
            _write_wav_16bit_dithered(wav_buf, samples, sr, dither_type=dither_type)
            wav_buf.seek(0)
            seg = AudioSegment.from_wav(wav_buf)
            out_buf = io.BytesIO()
            seg.export(out_buf, format="mp3", bitrate="320k")
            out_buf.seek(0)
            return out_buf.getvalue()
        except FileNotFoundError as exc:
            _raise_ffmpeg_error("mp3", exc)

    # Прочие форматы через pydub
    try:
        wav_buf = io.BytesIO()
        _write_wav_16bit_dithered(wav_buf, samples, sr, dither_type=dither_type)
        wav_buf.seek(0)
        seg = AudioSegment.from_wav(wav_buf)
        out_buf = io.BytesIO()
        seg.export(out_buf, format=out_format)
        out_buf.seek(0)
        return out_buf.getvalue()
    except FileNotFoundError as exc:
        _raise_ffmpeg_error(out_format, exc)


# Пресеты ревербератора: (decay_sec, comb_delays_ms, comb_gains, allpass_delays_ms, allpass_gains)
_REVERB_PRESETS = {
    "plate":  (1.2, [29, 37, 41, 53], [0.7, 0.65, 0.6, 0.55], [5, 7], [0.5, 0.4]),
    "room":   (0.6, [23, 31, 43, 47], [0.5, 0.45, 0.4, 0.35], [3, 5], [0.4, 0.3]),
    "hall":   (2.2, [47, 53, 61, 71], [0.75, 0.7, 0.65, 0.6], [8, 11], [0.5, 0.45]),
    "theater": (3.5, [59, 67, 73, 83], [0.78, 0.73, 0.68, 0.63], [10, 14], [0.52, 0.45]),
    "cathedral": (5.0, [97, 103, 109, 127], [0.82, 0.78, 0.74, 0.7], [15, 19], [0.55, 0.48]),
}


def _comb_filter(x: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    """Один comb-фильтр: y[n] = x[n] + gain * y[n - delay]."""
    out = np.zeros_like(x)
    if delay_samples <= 0 or delay_samples >= len(x):
        return x
    out[:delay_samples] = x[:delay_samples]
    for i in range(delay_samples, len(x)):
        out[i] = x[i] + gain * out[i - delay_samples]
    return out


def _allpass_filter(x: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    """Один allpass: y[n] = -gain*x[n] + x[n-delay] + gain*y[n-delay]."""
    out = np.zeros_like(x)
    if delay_samples <= 0 or delay_samples >= len(x):
        return x
    out[:delay_samples] = -gain * x[:delay_samples]
    for i in range(delay_samples, len(x)):
        out[i] = -gain * x[i] + x[i - delay_samples] + gain * out[i - delay_samples]
    return out


def _reverb_mono(
    x: np.ndarray,
    sr: int,
    reverb_type: str,
    decay_sec: float,
    mix: float,
) -> np.ndarray:
    """Один канал: dry*(1-mix) + wet*mix. x — моно массив."""
    preset = _REVERB_PRESETS.get(reverb_type, _REVERB_PRESETS["plate"])
    decay = decay_sec if decay_sec > 0 else preset[0]
    comb_delays_ms, comb_gains, ap_delays_ms, ap_gains = preset[1], preset[2], preset[3], preset[4]
    decay_per_sec = 0.001 ** (1.0 / max(0.1, decay))
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    wet = np.zeros(n)
    for d_ms, g in zip(comb_delays_ms, comb_gains):
        d_samp = min(int(sr * d_ms / 1000.0), n - 1)
        if d_samp < 1:
            continue
        g_adj = g * (decay_per_sec ** (d_ms / 1000.0))
        wet += _comb_filter(x, d_samp, g_adj)
    wet /= max(len(comb_delays_ms), 1)
    for d_ms, g in zip(ap_delays_ms, ap_gains):
        d_samp = min(int(sr * d_ms / 1000.0), n - 1)
        if d_samp < 1:
            continue
        wet = _allpass_filter(wet, d_samp, g)
    peak = np.max(np.abs(wet))
    if peak > 1e-6:
        wet = wet / min(peak, 2.0)
    return (x * (1.0 - mix) + wet * mix).astype(np.float32)


def apply_reverb(
    audio: np.ndarray,
    sr: int,
    reverb_type: str = "plate",
    decay_sec: float = 1.2,
    mix: float = 0.15,
    mix_mid: float | None = None,
    mix_side: float | None = None,
) -> np.ndarray:
    """
    Алгоритмический реверб (Schroeder: comb + allpass). reverb_type: plate, room, hall, theater, cathedral.
    mix: 0 = dry, 1 = full wet. mix_mid / mix_side: при стерео — отдельный mix для Mid и Side (M/S).
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    use_ms = (
        audio.shape[1] == 2
        and (mix_mid is not None or mix_side is not None)
    )
    if use_ms:
        mid_in = ((audio[:, 0] + audio[:, 1]) * 0.5).astype(np.float64)
        side_in = ((audio[:, 0] - audio[:, 1]) * 0.5).astype(np.float64)
        m_mid = float(mix_mid) if mix_mid is not None else mix
        m_side = float(mix_side) if mix_side is not None else mix
        m_mid = max(0.0, min(1.0, m_mid))
        m_side = max(0.0, min(1.0, m_side))
        mid_out = _reverb_mono(mid_in, sr, reverb_type, decay_sec, m_mid)
        side_out = _reverb_mono(side_in, sr, reverb_type, decay_sec, m_side)
        out_l = np.clip(mid_out + side_out, -1.0, 1.0).astype(np.float32)
        out_r = np.clip(mid_out - side_out, -1.0, 1.0).astype(np.float32)
        return np.stack([out_l, out_r], axis=1)
    preset = _REVERB_PRESETS.get(reverb_type, _REVERB_PRESETS["plate"])
    decay_sec = decay_sec if decay_sec > 0 else preset[0]
    comb_delays_ms, comb_gains, ap_delays_ms, ap_gains = preset[1], preset[2], preset[3], preset[4]
    decay_per_sec = 0.001 ** (1.0 / max(0.1, decay_sec))
    n = audio.shape[0]
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = _reverb_mono(
            audio[:, ch].astype(np.float64), sr, reverb_type, decay_sec, mix
        )
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    if audio.shape[1] == 1:
        return out[:, 0]
    return out


def _exciter_saturate(x: np.ndarray, mode: str, k: float = 2.0) -> np.ndarray:
    """
    Кривые насыщения по режиму эксайтера (Ozone 5-inspired).
    x — нормализованный сигнал; k — коэффициент жёсткости.
    """
    x = np.clip(x, -1.0, 1.0)
    if mode == "transistor":
        return x - (x ** 3) / 3.0
    if mode == "tape":
        eps = 1e-8
        return np.tanh(k * x) / (k + eps)
    if mode == "tube":
        alpha = 0.3
        return x + alpha * (x ** 2)
    if mode == "warm":
        return 0.5 * (np.tanh(k * x) / (k + 1e-8) + x + 0.3 * (x ** 2))
    if mode == "digital":
        return np.where(np.abs(x) <= 1.0, x, np.sign(x) * (2.0 - np.abs(x)))
    return np.tanh(k * x) / (k + 1e-8)


def apply_harmonic_exciter(
    audio: np.ndarray,
    sr: int,
    exciter_db: float = 0.0,
    mode: str = "warm",
    oversample: int = 1,
) -> np.ndarray:
    """
    Аналог iZotope Ozone 5 Exciter.dll — гармонический эксайтер.
    Добавляет гармоническое насыщение к высокочастотной части (> 3 kHz).
    mode: "warm" (по умолчанию), "tape", "tube", "transistor", "digital".
    oversample: 1, 2 или 4 — передискретизация перед насыщением для уменьшения алиасинга.
    """
    if abs(exciter_db) < 0.05:
        return audio
    from scipy import signal as sg

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    n_orig = audio.shape[0]
    n_ch = audio.shape[1]
    os = max(1, min(4, int(oversample)))
    if os <= 1:
        work_sr = sr
        work = audio
    else:
        n_high = n_orig * os
        work = np.zeros((n_high, n_ch), dtype=np.float32)
        for c in range(n_ch):
            work[:, c] = sg.resample(audio[:, c].astype(np.float64), n_high).astype(np.float32)
        work_sr = sr * os

    nyq = work_sr / 2.0
    f_lo = min(3000.0 / nyq, 0.97)
    b_hp, a_hp = sg.butter(2, f_lo, btype="high", output="ba")
    gain = 10 ** (exciter_db / 20.0) - 1.0
    valid_modes = ("warm", "tape", "tube", "transistor", "digital")
    sat_mode = mode if mode in valid_modes else "warm"
    k = 2.5 if sat_mode == "warm" else 2.0
    out_work = work.copy()
    for ch in range(work.shape[1]):
        hf = sg.filtfilt(b_hp, a_hp, work[:, ch])
        saturated = _exciter_saturate(hf, sat_mode, k)
        excitation = (saturated - hf) * gain * 0.35
        out_work[:, ch] = work[:, ch] + excitation

    if os > 1:
        out = np.zeros((n_orig, n_ch), dtype=np.float32)
        for c in range(n_ch):
            out[:, c] = sg.resample(out_work[:, c].astype(np.float64), n_orig).astype(np.float32)
    else:
        out = out_work.astype(np.float32)

    if n_ch == 1:
        return out[:, 0]
    return out


def _imager_apply_width_stereo(left: np.ndarray, right: np.ndarray, width: float) -> tuple:
    """M/S: scale side by width, return (out_l, out_r)."""
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    side_scaled = side * width
    out_l = np.clip(mid + side_scaled, -1.0, 1.0)
    out_r = np.clip(mid - side_scaled, -1.0, 1.0)
    return out_l, out_r


def apply_stereo_imager(
    audio: np.ndarray,
    width: float = 1.0,
    stereoize_delay_ms: float = 0.0,
    stereoize_mix: float = 0.12,
    sr: int | None = None,
    band_widths: tuple[float, float, float, float] | list[float] | None = None,
    crossovers_hz: tuple[float, float, float] | list[float] | None = None,
) -> np.ndarray:
    """
    Регулятор стереобазы (Mid-Side). width: 0 = моно, 1 = оригинал, >1 = шире.
    stereoize_delay_ms: 0 = выкл, 1–15 ms — Haas-задержка для синтетического расширения (требуется sr).
    stereoize_mix: уровень кросс-задержки (0.05–0.25).
    band_widths: опционально (w0,w1,w2,w3) — width по 4 полосам; при задании используется 4-band режим.
    crossovers_hz: кроссоверы для 4-band (по умолчанию 214, 2230, 10000 Hz).
    """
    if audio.ndim == 1 or audio.shape[1] == 1:
        return audio
    left = audio[:, 0].astype(np.float32)
    right = audio[:, 1].astype(np.float32)

    if band_widths is not None and len(band_widths) == 4 and sr and sr > 0:
        # 4-band: split stereo into bands, apply width per band, merge
        cross = (
            tuple(float(x) for x in crossovers_hz)
            if crossovers_hz and len(crossovers_hz) == 3
            else MULTIBAND_CROSSOVERS_HZ
        )
        cross = tuple(np.clip(c, 20.0, 20000.0) for c in cross)
        if cross[0] >= cross[1] or cross[1] >= cross[2]:
            cross = MULTIBAND_CROSSOVERS_HZ
        stereo = np.column_stack([left, right])
        bands = _split_bands(stereo, float(sr), cross)
        widths = tuple(float(band_widths[i]) for i in range(4))
        out_l = np.zeros_like(left)
        out_r = np.zeros_like(right)
        for i in range(4):
            b = bands[i]
            if b.ndim == 1:
                bl, br = b, b
            else:
                bl, br = b[:, 0], b[:, 1]
            ol, or_ = _imager_apply_width_stereo(bl, br, widths[i])
            out_l += ol
            out_r += or_
        out_l = np.clip(out_l, -1.0, 1.0)
        out_r = np.clip(out_r, -1.0, 1.0)
    else:
        out_l, out_r = _imager_apply_width_stereo(left, right, width)

    if stereoize_delay_ms > 0 and sr and sr > 0 and stereoize_mix > 0:
        delay_n = min(int(sr * stereoize_delay_ms / 1000.0), audio.shape[0] - 1)
        delay_n = max(0, delay_n)
        mix = min(0.35, max(0.0, float(stereoize_mix)))
        if delay_n > 0:
            delayed_r = np.concatenate([np.zeros(delay_n, dtype=out_r.dtype), out_r[:-delay_n]])
            delayed_l = np.concatenate([np.zeros(delay_n, dtype=out_l.dtype), out_l[:-delay_n]])
            out_l = np.clip(out_l + mix * delayed_r, -1.0, 1.0)
            out_r = np.clip(out_r + mix * delayed_l, -1.0, 1.0)
    return np.column_stack([out_l, out_r]).astype(np.float32)


def apply_style_eq(audio: np.ndarray, sr: int, style: str = "standard") -> np.ndarray:
    """
    Жанровый EQ: мягкие буст/срез по 5 полосам поверх основного конвейера.
    Применяется последним шагом перед финальным лимитером.
    """
    from scipy import signal as sg
    cfg = STYLE_CONFIGS.get(style, STYLE_CONFIGS["standard"])
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    nyq = sr / 2.0
    # (f_lo, f_hi, gain_db)
    bands = [
        (30.0,   90.0,   cfg["sub"]),
        (90.0,  280.0,   cfg["bass"]),
        (700.0, 2800.0,  cfg["mids"]),
        (2800.0, 9000.0, cfg["presence"]),
        (10000.0, min(sr * 0.46, 18000.0), cfg["air"]),
    ]
    out = audio.copy().astype(np.float32)
    for f_lo, f_hi, gain_db in bands:
        if abs(gain_db) < 0.05:
            continue
        f_lo_n = min(f_lo / nyq, 0.98)
        f_hi_n = min(f_hi / nyq, 0.98)
        if f_lo_n >= f_hi_n:
            continue
        b, a = sg.butter(1, [f_lo_n, f_hi_n], btype="band", output="ba")
        g = 10 ** (gain_db / 20.0)
        for ch in range(out.shape[1]):
            band = sg.filtfilt(b, a, out[:, ch])
            out[:, ch] = out[:, ch] + (g - 1.0) * band
    if audio.shape[1] == 1:
        return out[:, 0]
    return out


def run_mastering_pipeline(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    style: str = "standard",
    progress_callback=None,
) -> np.ndarray:
    """
    Конвейер студийного мастеринга на основе модулей iZotope Ozone 5.
    Для стиля house_basic дополнительно применяются:
      — apply_harmonic_exciter (аналог Ozone 5 Exciter.dll)
      — apply_stereo_imager    (аналог Ozone 5 Imager.dll)
    progress_callback(percent: int, message: str) вызывается на каждом шаге (0–100).
    """
    def report(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    cfg = STYLE_CONFIGS.get(style, STYLE_CONFIGS["standard"])
    exciter_db    = cfg.get("exciter_db", 0.0)
    imager_width  = cfg.get("imager_width", 1.0)

    report(5, "Подготовка…")
    audio = remove_dc_offset(audio)
    report(12, "Удаление DC-смещения")
    audio = remove_intersample_peaks(audio, headroom_db=0.5)
    report(20, "Защита от пиков")
    audio = apply_target_curve(audio, sr)
    report(35, "Студийный EQ (Ozone 5 Equalizer)")
    audio = apply_dynamics(audio, sr)
    report(55, "Многополосная динамика и максимайзер (Ozone 5 Dynamics / Maximizer)")
    audio = normalize_lufs(audio, sr, target_lufs)
    report(70, "Нормализация LUFS")
    audio = apply_final_spectral_balance(audio, sr)
    report(78, "Финальная частотная коррекция")
    audio = apply_style_eq(audio, sr, style)
    report(84, f"Жанровый EQ · {style}")

    if exciter_db > 0.05:
        audio = apply_harmonic_exciter(audio, sr, exciter_db)
        report(88, f"Гармонический эксайтер (Ozone 5 Exciter) · +{exciter_db:.1f} dB")

    if abs(imager_width - 1.0) > 0.01:
        audio = apply_stereo_imager(audio, imager_width)
        report(91, f"Стерео-расширение (Ozone 5 Imager) · width={imager_width:.2f}")

    audio = remove_intersample_peaks(audio, headroom_db=0.5)
    report(94, "Финальная защита пиков")
    out = np.clip(audio, -1.0, 1.0).astype(np.float32)
    out = np.ascontiguousarray(out)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    report(95, "Готово")
    return out
