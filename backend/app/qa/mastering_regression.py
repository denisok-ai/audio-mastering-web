"""
Офлайн-метрики по временным окнам для регрессии мастеринга (v2 default chain).
Используется тестами и скриптами; не импортировать из горячего пути API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Окна из жалобы пользователя (сек): вступление, ~1:15–1:30, ~2:34–2:40
DEFAULT_WINDOWS_SEC: tuple[tuple[str, float, float], ...] = (
    ("intro", 2.0, 10.0),
    ("mid_75_90", 75.0, 90.0),
    ("late_154_160", 154.0, 160.0),
)


def regression_wav_path() -> Path | None:
    """Путь к WAV: MM_REGRESSION_WAV или фикстура по умолчанию / test_output в корне репозитория."""
    env = os.environ.get("MM_REGRESSION_WAV", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    backend_root = Path(__file__).resolve().parent.parent.parent
    repo_root = backend_root.parent
    candidates = (
        backend_root / "tests" / "fixtures" / "mastering_regression" / "alors_on_danse_rem.wav",
        backend_root / "tests" / "fixtures" / "mastering_regression" / "_Alors On Danse Rem.wav",
        repo_root / "test_output" / "_Alors On Danse Rem.wav",
        repo_root / "test_output" / "alors_on_danse_rem.wav",
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_fixture_metrics_path() -> Path | None:
    p = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "mastering_regression" / "expected_metrics.json"
    return p if p.is_file() else None


def to_mono_float64(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a
    return np.mean(a, axis=1)


def slice_window(mono: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(t0 * sr))
    i1 = min(len(mono), int(t1 * sr))
    if i0 >= i1:
        return mono[:0]
    return mono[i0:i1]


def hf_rms(mono_win: np.ndarray, sr: int, hp_hz: float = 8000.0) -> float:
    if mono_win.size < 32:
        return 0.0
    from scipy import signal as sg

    nyq = sr / 2.0
    wn = min(float(hp_hz) / nyq, 0.99)
    b, a = sg.butter(2, wn, btype="high", output="ba")
    try:
        hf = sg.filtfilt(b, a, mono_win)
    except Exception:  # noqa: BLE001
        hf = mono_win
    return float(np.sqrt(np.mean(hf * hf) + 1e-20))


def max_abs_first_diff(mono_win: np.ndarray) -> float:
    if mono_win.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(mono_win))))


def window_metrics(audio: np.ndarray, sr: int, windows_sec: Iterable[tuple[str, float, float]] = DEFAULT_WINDOWS_SEC) -> dict[str, dict[str, float]]:
    mono = to_mono_float64(audio)
    out: dict[str, dict[str, float]] = {}
    for name, t0, t1 in windows_sec:
        w = slice_window(mono, sr, t0, t1)
        out[name] = {
            "hf_rms": hf_rms(w, sr),
            "max_abs_diff": max_abs_first_diff(w),
            "rms": float(np.sqrt(np.mean(w * w) + 1e-20)) if w.size else 0.0,
            "samples": float(w.size),
        }
    return out


def run_default_chain_stages(
    audio: np.ndarray,
    sr: int,
    *,
    target_lufs: float = -14.0,
    style: str = "standard",
) -> list[tuple[str, np.ndarray]]:
    """Как v2 без PRO: каждый модуль default_chain + финальный clip (как MasteringChain.process)."""
    from app.chain import MasteringChain

    chain = MasteringChain.default_chain(target_lufs=target_lufs, style=style)
    stages: list[tuple[str, np.ndarray]] = []
    a = np.asarray(audio, dtype=np.float32)
    for mod in chain.modules:
        kw: dict[str, Any] = {"target_lufs": target_lufs, "style": style}
        a = mod.process(a, sr, **kw)
        stages.append((getattr(mod, "module_id", "?"), np.copy(a)))
    a = np.ascontiguousarray(np.clip(a, -1.0, 1.0).astype(np.float32))
    np.nan_to_num(a, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    stages.append(("chain_finalize_clip", np.copy(a)))
    from app.pipeline import apply_output_edge_fade_in

    faded = apply_output_edge_fade_in(a, sr, fade_ms=6.0)
    stages.append(("v2_output_fade_in", np.copy(faded)))
    return stages


def metrics_after_each_stage(
    audio: np.ndarray,
    sr: int,
    windows_sec: Iterable[tuple[str, float, float]] = DEFAULT_WINDOWS_SEC,
    **chain_kw: Any,
) -> list[dict[str, Any]]:
    """Список {stage, windows: {name: metrics}} для сопоставления с логами prod (stage / module_id)."""
    rows: list[dict[str, Any]] = []
    for stage_id, buf in run_default_chain_stages(audio, sr, **chain_kw):
        rows.append(
            {
                "stage": stage_id,
                "windows": window_metrics(buf, sr, windows_sec),
            }
        )
    return rows


def load_expected_thresholds() -> dict[str, Any] | None:
    p = load_fixture_metrics_path()
    if not p:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
