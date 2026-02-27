# @file equalizer.py
# @description Модули EQ: Target Curve, Final Spectral Balance, Style EQ
from typing import Any
import numpy as np
from .base import BaseModule
from ..pipeline import (
    apply_target_curve,
    apply_final_spectral_balance,
    apply_style_eq,
)


class TargetCurveModule(BaseModule):
    module_id = "target_curve"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        ms_mode: str = "both",
        phase_mode: str = "minimum",
        eq_ms: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            enabled=enabled,
            amount=amount,
            ms_mode=ms_mode,
            phase_mode=phase_mode,
            eq_ms=eq_ms,
            **kwargs,
        )
        self.phase_mode = str(self.params.get("phase_mode", phase_mode))
        self.eq_ms = bool(self.params.get("eq_ms", eq_ms))

    @classmethod
    def from_config(cls, config: dict) -> "TargetCurveModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            ms_mode=str(config.get("ms_mode", "both")),
            phase_mode=str(config.get("phase_mode", "minimum")),
            eq_ms=bool(config.get("eq_ms", False)),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        phase_mode = kwargs.get("phase_mode", self.phase_mode)
        eq_ms = kwargs.get("eq_ms", self.eq_ms)
        return apply_target_curve(audio, sr, phase_mode=phase_mode, eq_ms=eq_ms)


class FinalSpectralBalanceModule(BaseModule):
    module_id = "final_spectral_balance"

    def __init__(self, enabled: bool = True, amount: float = 1.0, **kwargs: Any):
        super().__init__(enabled=enabled, amount=amount, **kwargs)

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return apply_final_spectral_balance(audio, sr)


class StyleEQModule(BaseModule):
    module_id = "style_eq"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        style: str = "standard",
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, style=style, **kwargs)
        self.style = str(self.params.get("style", style))

    @classmethod
    def from_config(cls, config: dict) -> "StyleEQModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            style=str(config.get("style", "standard")),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        style = kwargs.get("style", self.style)
        return apply_style_eq(audio, sr, style=style)
