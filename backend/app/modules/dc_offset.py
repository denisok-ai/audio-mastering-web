# @file dc_offset.py
# @description Модуль удаления DC-смещения
from typing import Any
import numpy as np
from .base import BaseModule
from ..pipeline import remove_dc_offset


class DCOffsetModule(BaseModule):
    module_id = "dc_offset"

    def __init__(self, enabled: bool = True, amount: float = 1.0, **kwargs: Any):
        super().__init__(enabled=enabled, amount=amount, **kwargs)

    def _process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> np.ndarray:
        return remove_dc_offset(audio)
