# @file chain.py
# @description Цепочка мастеринга из конфига (v2)
# @created 2026-02-27

from typing import Any, Callable, Optional

import numpy as np

from .modules.base import BaseModule
from .modules.dc_offset import DCOffsetModule
from .modules.dynamics import DynamicsModule
from .modules.equalizer import FinalSpectralBalanceModule, StyleEQModule, TargetCurveModule
from .modules.exciter import ExciterModule
from .modules.imaging import ImagerModule
from .modules.maximizer import MaximizerModule
from .modules.normalize_lufs import NormalizeLUFSModule
from .modules.peak_guard import PeakGuardModule
from .modules.reverb import ReverbModule

# Регистр: module_id -> класс модуля
MODULE_REGISTRY: dict[str, type[BaseModule]] = {
    DCOffsetModule.module_id: DCOffsetModule,
    PeakGuardModule.module_id: PeakGuardModule,
    TargetCurveModule.module_id: TargetCurveModule,
    DynamicsModule.module_id: DynamicsModule,
    MaximizerModule.module_id: MaximizerModule,
    NormalizeLUFSModule.module_id: NormalizeLUFSModule,
    FinalSpectralBalanceModule.module_id: FinalSpectralBalanceModule,
    StyleEQModule.module_id: StyleEQModule,
    ExciterModule.module_id: ExciterModule,
    ImagerModule.module_id: ImagerModule,
    ReverbModule.module_id: ReverbModule,
}


class MasteringChain:
    """
    Цепочка модулей мастеринга. Собирается из JSON-конфига.
    process(audio, sr, ...) последовательно применяет все включённые модули.
    """

    def __init__(self, modules: list[BaseModule]):
        self.modules = modules

    @classmethod
    def from_config(cls, config: dict) -> "MasteringChain":
        """
        config["modules"] — список словарей {"id": "...", "enabled": true, ...}.
        Остальные поля config (target_lufs, style) передаются в kwargs при process().
        """
        modules: list[BaseModule] = []
        for item in config.get("modules", []):
            item = dict(item)
            mid = item.pop("id", None)
            if not mid or mid not in MODULE_REGISTRY:
                continue
            mod_cls = MODULE_REGISTRY[mid]
            mod = mod_cls.from_config(item)
            modules.append(mod)
        return cls(modules=modules)

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        target_lufs: Optional[float] = None,
        style: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Последовательно применяет все модули цепочки.
        target_lufs и style передаются в модули через kwargs.
        """
        total = len(self.modules)
        for i, mod in enumerate(self.modules):
            if progress_callback and total > 0:
                pct = 5 + int(90 * (i / total))
                progress_callback(pct, getattr(mod, "module_id", "module"))
            kw = dict(kwargs)
            if target_lufs is not None:
                kw["target_lufs"] = target_lufs
            if style is not None:
                kw["style"] = style
            audio = mod.process(audio, sr, **kw)
        audio = np.ascontiguousarray(np.clip(audio, -1.0, 1.0).astype(np.float32))
        np.nan_to_num(audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        if progress_callback:
            progress_callback(98, "Готово")
        return audio

    @classmethod
    def default_config(cls, target_lufs: float = -14.0, style: str = "standard") -> dict:
        """
        Конфиг цепочки по умолчанию (тот же, что передаётся в from_config).
        Для отдачи в API и последующей отправки в POST /api/v2/master (в т.ч. с изменённым порядком).
        """
        from .pipeline import STYLE_CONFIGS

        cfg = STYLE_CONFIGS.get(style, STYLE_CONFIGS["standard"])
        exciter_db = cfg.get("exciter_db", 0.0)
        imager_width = cfg.get("imager_width", 1.0)
        return {
            "modules": [
                {"id": "dc_offset", "enabled": True, "amount": 1.0},
                {"id": "peak_guard", "enabled": True, "headroom_db": 0.5, "amount": 1.0},
                {"id": "target_curve", "enabled": True, "phase_mode": "minimum", "eq_ms": False, "amount": 1.0},
                {"id": "dynamics", "enabled": True, "knee_db": 6.0, "crossovers_hz": [214.0, 2230.0, 10000.0], "amount": 1.0},
                {"id": "normalize_lufs", "enabled": True, "target_lufs": target_lufs, "amount": 1.0},
                {"id": "final_spectral_balance", "enabled": True, "amount": 1.0},
                {"id": "style_eq", "enabled": True, "style": style, "amount": 1.0},
                {"id": "exciter", "enabled": abs(exciter_db) >= 0.05, "exciter_db": exciter_db, "mode": "warm", "oversample": 1, "amount": 1.0},
                {"id": "imager", "enabled": abs(imager_width - 1.0) >= 0.01, "width": imager_width, "stereoize_delay_ms": 0.0, "stereoize_mix": 0.12, "band_widths": None, "crossovers_hz": [214.0, 2230.0, 10000.0], "amount": 1.0},
                {"id": "reverb", "enabled": False, "reverb_type": "plate", "decay_sec": 1.2, "mix": 0.15, "mix_mid": None, "mix_side": None, "amount": 1.0},
                {"id": "peak_guard", "enabled": True, "headroom_db": 0.5, "amount": 1.0},
            ]
        }

    @classmethod
    def default_chain(cls, target_lufs: float = -14.0, style: str = "standard") -> "MasteringChain":
        """
        Цепочка по умолчанию (эквивалент run_mastering_pipeline v1).
        Параметры exciter_db и imager_width берутся из STYLE_CONFIGS[style].
        """
        config = cls.default_config(target_lufs=target_lufs, style=style)
        return cls.from_config(config)
