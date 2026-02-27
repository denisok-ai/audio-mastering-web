# @file imaging.py
# @description Модуль стерео-расширения (аналог iZotope Ozone 5 Imager)
from typing import Any, Optional
import numpy as np
from .base import BaseModule
from ..pipeline import apply_stereo_imager


class ImagerModule(BaseModule):
    module_id = "imager"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        width: float = 1.0,
        stereoize_delay_ms: float = 0.0,
        stereoize_mix: float = 0.12,
        band_widths: Optional[list] = None,
        crossovers_hz: Optional[list] = None,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.width = float(self.params.get("width", width))
        self.stereoize_delay_ms = float(self.params.get("stereoize_delay_ms", stereoize_delay_ms))
        self.stereoize_mix = float(self.params.get("stereoize_mix", stereoize_mix))
        raw_bw = self.params.get("band_widths", band_widths)
        self.band_widths = list(raw_bw) if raw_bw else None
        raw_cross = self.params.get("crossovers_hz", crossovers_hz)
        self.crossovers_hz = tuple(float(x) for x in raw_cross) if raw_cross else None

    @classmethod
    def from_config(cls, config: dict) -> "ImagerModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            width=float(config.get("width", 1.0)),
            stereoize_delay_ms=float(config.get("stereoize_delay_ms", 0.0)),
            stereoize_mix=float(config.get("stereoize_mix", 0.12)),
            band_widths=config.get("band_widths"),
            crossovers_hz=config.get("crossovers_hz"),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return apply_stereo_imager(
            audio,
            width=self.width,
            stereoize_delay_ms=self.stereoize_delay_ms,
            stereoize_mix=self.stereoize_mix,
            sr=sr,
            band_widths=self.band_widths,
            crossovers_hz=self.crossovers_hz,
        )
