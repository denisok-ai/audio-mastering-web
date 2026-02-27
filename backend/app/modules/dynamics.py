# @file dynamics.py
# @description Модуль многополосной динамики
from typing import Any, Optional
import numpy as np
from .base import BaseModule
from ..pipeline import apply_dynamics


class DynamicsModule(BaseModule):
    module_id = "dynamics"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        knee_db: float = 6.0,
        crossovers_hz: Optional[list] = None,
        band_ratios: Optional[list] = None,
        max_upward_boost_db: float = 12.0,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.knee_db = float(self.params.get("knee_db", knee_db))
        raw_cross = self.params.get("crossovers_hz", crossovers_hz)
        self.crossovers_hz = tuple(float(x) for x in raw_cross) if raw_cross else None
        raw_br = self.params.get("band_ratios", band_ratios)
        self.band_ratios = tuple(float(x) for x in raw_br) if raw_br else None
        self.max_upward_boost_db = float(self.params.get("max_upward_boost_db", max_upward_boost_db))

    @classmethod
    def from_config(cls, config: dict) -> "DynamicsModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            knee_db=float(config.get("knee_db", 6.0)),
            crossovers_hz=config.get("crossovers_hz"),
            band_ratios=config.get("band_ratios"),
            max_upward_boost_db=float(config.get("max_upward_boost_db", 12.0)),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return apply_dynamics(
            audio,
            sr,
            knee_db=self.knee_db,
            crossovers_hz=self.crossovers_hz,
            band_ratios=self.band_ratios,
            max_upward_boost_db=self.max_upward_boost_db,
        )
