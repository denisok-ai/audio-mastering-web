# @file reverb.py
# @description Модуль ревербератора (Schroeder plate/room/hall/theater/cathedral)
from typing import Any, Optional
import numpy as np
from .base import BaseModule
from ..pipeline import apply_reverb


class ReverbModule(BaseModule):
    module_id = "reverb"

    def __init__(
        self,
        enabled: bool = False,
        amount: float = 1.0,
        reverb_type: str = "plate",
        decay_sec: float = 1.2,
        mix: float = 0.15,
        mix_mid: Optional[float] = None,
        mix_side: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.reverb_type = str(self.params.get("reverb_type", reverb_type))
        self.decay_sec = float(self.params.get("decay_sec", decay_sec))
        self.mix = float(self.params.get("mix", mix))
        mm = self.params.get("mix_mid", mix_mid)
        ms = self.params.get("mix_side", mix_side)
        self.mix_mid = float(mm) if mm is not None else None
        self.mix_side = float(ms) if ms is not None else None

    @classmethod
    def from_config(cls, config: dict) -> "ReverbModule":
        mm = config.get("mix_mid")
        ms = config.get("mix_side")
        return cls(
            enabled=bool(config.get("enabled", False)),
            amount=float(config.get("amount", 1.0)),
            reverb_type=str(config.get("reverb_type", "plate")),
            decay_sec=float(config.get("decay_sec", 1.2)),
            mix=float(config.get("mix", 0.15)),
            mix_mid=float(mm) if mm is not None else None,
            mix_side=float(ms) if ms is not None else None,
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return apply_reverb(
            audio, sr,
            reverb_type=self.reverb_type,
            decay_sec=self.decay_sec,
            mix=self.mix,
            mix_mid=self.mix_mid,
            mix_side=self.mix_side,
        )
