# @file peak_guard.py
# @description Модуль защиты от пиков (headroom limiter)
from typing import Any
import numpy as np
from .base import BaseModule
from ..pipeline import remove_intersample_peaks


class PeakGuardModule(BaseModule):
    module_id = "peak_guard"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        headroom_db: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.headroom_db = float(self.params.get("headroom_db", headroom_db))

    @classmethod
    def from_config(cls, config: dict) -> "PeakGuardModule":
        headroom_db = float(config.get("headroom_db", 0.5))
        enabled = bool(config.get("enabled", True))
        amount = float(config.get("amount", 1.0))
        return cls(enabled=enabled, amount=amount, headroom_db=headroom_db)

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return remove_intersample_peaks(audio, headroom_db=self.headroom_db)
