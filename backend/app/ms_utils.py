# @file ms_utils.py
# @description Mid-Side encode/decode для стерео-обработки
# @dependencies numpy
# @created 2026-02-27

import numpy as np


def mid_side_encode(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Стерео (L, R) → Mid (M) и Side (S).
    M = (L + R) / 2,  S = (L - R) / 2.
    audio: (samples, 2). Возвращает (mid, side) — оба (samples,).
    """
    if audio.ndim == 1 or audio.shape[1] != 2:
        raise ValueError("mid_side_encode ожидает стерео (samples, 2)")
    left = np.asarray(audio[:, 0], dtype=np.float32)
    right = np.asarray(audio[:, 1], dtype=np.float32)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    return mid, side


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """
    Mid (M) и Side (S) → стерео (L, R).
    L = M + S,  R = M - S.
    Возвращает (samples, 2).
    """
    mid = np.asarray(mid, dtype=np.float32)
    side = np.asarray(side, dtype=np.float32)
    left = np.clip(mid + side, -1.0, 1.0)
    right = np.clip(mid - side, -1.0, 1.0)
    return np.column_stack([left, right])
