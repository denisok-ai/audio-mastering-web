# @file exciter.py
# @description Модуль гармонического эксайтера (аналог iZotope Ozone 5 Exciter)
from typing import Any
import numpy as np
from .base import BaseModule
from ..pipeline import apply_harmonic_exciter


class ExciterModule(BaseModule):
    module_id = "exciter"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        exciter_db: float = 0.0,
        mode: str = "warm",
        oversample: int = 1,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.exciter_db = float(self.params.get("exciter_db", exciter_db))
        self.mode = str(self.params.get("mode", mode))
        self.oversample = int(self.params.get("oversample", oversample))

    @classmethod
    def from_config(cls, config: dict) -> "ExciterModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            exciter_db=float(config.get("exciter_db", 0.0)),
            mode=str(config.get("mode", "warm")),
            oversample=int(config.get("oversample", 1)),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return apply_harmonic_exciter(
            audio, sr,
            exciter_db=self.exciter_db,
            mode=self.mode,
            oversample=self.oversample,
        )
