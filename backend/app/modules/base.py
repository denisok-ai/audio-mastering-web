# @file base.py
# @description Базовый класс модуля мастеринга
from typing import Any
import numpy as np


class BaseModule:
    """
    Базовый класс для всех модулей цепочки мастеринга.
    Каждый модуль имеет module_id, флаг enabled и amount (0–1 blend с оригиналом).
    """

    module_id: str = "base"

    def __init__(
        self,
        enabled: bool = True,
        amount: float = 1.0,
        ms_mode: str = "both",
        **kwargs: Any,
    ):
        self.enabled = bool(enabled)
        self.amount = float(np.clip(amount, 0.0, 1.0))
        self.ms_mode = str(ms_mode)
        self.params = kwargs

    @classmethod
    def from_config(cls, config: dict) -> "BaseModule":
        """Создать модуль из словаря параметров."""
        return cls(**config)

    def process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        """
        Применить модуль к аудио.
        Если disabled — вернуть без изменений.
        amount < 1 — линейный blend processed/original.
        """
        if not self.enabled:
            return audio
        try:
            processed = self._process(audio, sr, **kwargs)
        except Exception:
            return audio
        if self.amount >= 1.0:
            return processed
        return (audio * (1.0 - self.amount) + processed * self.amount).astype(np.float32)

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        """Переопределить в подклассе."""
        return audio
