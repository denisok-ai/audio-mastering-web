# @file normalize_lufs.py
# @description Модуль нормализации LUFS
from typing import Any
import numpy as np
from .base import BaseModule
from ..pipeline import normalize_lufs


class NormalizeLUFSModule(BaseModule):
    module_id = "normalize_lufs"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        target_lufs: float = -14.0,
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled, amount=amount, **kwargs)
        self.target_lufs = float(self.params.get("target_lufs", target_lufs))

    @classmethod
    def from_config(cls, config: dict) -> "NormalizeLUFSModule":
        return cls(
            enabled=bool(config.get("enabled", True)),
            amount=float(config.get("amount", 1.0)),
            target_lufs=float(config.get("target_lufs", -14.0)),
        )

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        target = kwargs.get("target_lufs", self.target_lufs)
        return normalize_lufs(audio, sr, target_lufs=float(target))
