# @file maximizer.py
# @description Модуль максимайзера (transient-aware)
from typing import Any
import numpy as np
from .base import BaseModule
from ..pipeline import apply_maximizer_transient_aware


class MaximizerModule(BaseModule):
    module_id = "maximizer"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        sensitivity: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.sensitivity = float(self.params.get("sensitivity", sensitivity))

    @classmethod
    def from_config(cls, config: dict) -> "MaximizerModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            sensitivity=float(config.get("sensitivity", 0.5)),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return apply_maximizer_transient_aware(audio, sr, sensitivity=self.sensitivity)
