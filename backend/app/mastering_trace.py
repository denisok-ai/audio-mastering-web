# @file mastering_trace.py
# @description Структурированные логи мастеринга при MAGIC_MASTER_MASTERING_TRACE=1

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .config import settings

# Имя логгера под пакет app.* — иначе uvicorn/journald на prod не показывают записи (не префикс app.).
_LOG = logging.getLogger(__name__)

# Uvicorn не вешает StreamHandler на app.* — записи не доходят до stderr/journald (см. app.bot.lifecycle).
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _LOG.addHandler(_h)
_LOG.setLevel(logging.INFO)

# Допускаем включение без перезапуска через env (как MAGIC_MASTER_DEBUG)
_ENV_TRACE = "MAGIC_MASTER_MASTERING_TRACE"
_ENV_LUFS = "MAGIC_MASTER_MASTERING_TRACE_LUFS_STAGES"


def _trace_enabled() -> bool:
    if getattr(settings, "mastering_trace", False):
        return True
    return os.environ.get(_ENV_TRACE, "").strip().lower() in ("1", "true", "yes", "on")


def _lufs_stages_enabled() -> bool:
    if getattr(settings, "mastering_trace_lufs_stages", False):
        return True
    return os.environ.get(_ENV_LUFS, "").strip().lower() in ("1", "true", "yes", "on")


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    base = os.path.basename(name or "unknown")
    base = re.sub(r"[^\w.\-]+", "_", base, flags=re.UNICODE)
    return base[:max_len] if len(base) > max_len else base


@dataclass
class TraceContext:
    job_id: str
    filename: str
    path: str  # v1 | v2 | telegram
    style: str = "standard"
    user_id: Optional[int] = None
    target_lufs: Optional[float] = None
    pro_flags: str = ""

    @staticmethod
    def build(
        job_id: str,
        filename: str,
        path: str,
        *,
        style: str = "standard",
        user_id: Optional[int] = None,
        target_lufs: Optional[float] = None,
        pro_params: Optional[Mapping[str, Any]] = None,
    ) -> TraceContext:
        flags = ""
        if pro_params:
            truthy: list[str] = []
            for k, v in sorted(pro_params.items()):
                if v is None or v is False or v == "":
                    continue
                if isinstance(v, bool) and v:
                    truthy.append(k)
                elif isinstance(v, (int, float)) and float(v) > 0.01 and k in (
                    "denoise_strength",
                    "parallel_mix",
                ):
                    truthy.append(k)
                elif k in (
                    "rumble_enabled",
                    "deesser_enabled",
                    "dynamic_eq_enabled",
                    "apply_vocal_isolation",
                ) and v not in (False, 0, 0.0, ""):
                    truthy.append(k)
            flags = ",".join(truthy[:24])
        return TraceContext(
            job_id=job_id,
            filename=_sanitize_filename(filename),
            path=path,
            style=(style or "standard")[:64],
            user_id=user_id,
            target_lufs=target_lufs,
            pro_flags=flags,
        )


def _fmt_kv(**kwargs: Any) -> str:
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def signal_metrics(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Быстрые метрики без LUFS."""
    a = np.asarray(audio)
    if a.size == 0:
        return {
            "channels": 0,
            "samples": 0,
            "duration_sec": 0.0,
            "peak_linear": 0.0,
            "peak_db": -120.0,
            "nan_count": 0,
            "inf_count": 0,
        }
    if a.ndim == 1:
        channels = 1
        n = int(a.shape[0])
    else:
        channels = int(a.shape[1])
        n = int(a.shape[0])
    finite = np.isfinite(a)
    nan_count = int(np.sum(~finite))
    inf_mask = np.isinf(a)
    inf_count = int(np.sum(inf_mask))
    with np.errstate(divide="ignore"):
        peak = float(np.max(np.abs(a[np.isfinite(a)]))) if np.any(finite) else 0.0
    peak_db = float(20.0 * np.log10(max(peak, 1e-12)))
    return {
        "channels": channels,
        "samples": n,
        "duration_sec": round(n / float(sr), 4) if sr else 0.0,
        "peak_linear": round(peak, 6),
        "peak_db": round(peak_db, 2),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def trace_stage(
    ctx: Optional[TraceContext],
    stage: str,
    audio: np.ndarray,
    sr: int,
    **extra: Any,
) -> None:
    if ctx is None or not _trace_enabled():
        return
    m = signal_metrics(audio, sr)
    payload = {
        "job_id": ctx.job_id,
        "path": ctx.path,
        "filename": ctx.filename,
        "stage": stage,
        **m,
        **{k: v for k, v in extra.items() if v is not None},
    }
    if _lufs_stages_enabled():
        try:
            from .pipeline import measure_lufs

            lu = measure_lufs(audio, sr)
            if lu == lu:
                payload["lufs"] = round(float(lu), 3)
        except Exception:  # noqa: BLE001
            payload["lufs"] = "err"
    _LOG.info("mastering_trace %s", _fmt_kv(**payload))


def trace_job_start(ctx: TraceContext) -> None:
    if not _trace_enabled():
        return
    d = asdict(ctx)
    _LOG.info("mastering_trace_start %s", _fmt_kv(**d))


def trace_job_done(
    ctx: TraceContext,
    *,
    before_lufs: Optional[float],
    after_lufs: Optional[float],
    out_format: str,
) -> None:
    if not _trace_enabled():
        return
    _LOG.info(
        "mastering_trace_done %s",
        _fmt_kv(
            job_id=ctx.job_id,
            path=ctx.path,
            filename=ctx.filename,
            before_lufs=before_lufs,
            after_lufs=after_lufs,
            out_format=out_format,
        ),
    )


def trace_job_error(ctx: TraceContext, exc: BaseException) -> None:
    if not _trace_enabled():
        return
    _LOG.info(
        "mastering_trace_error %s",
        _fmt_kv(
            job_id=ctx.job_id,
            path=ctx.path,
            filename=ctx.filename,
            error_type=type(exc).__name__,
            error_msg=str(exc)[:500],
        ),
    )


def trace_chain_modules(ctx: Optional[TraceContext], module_ids: list[str]) -> None:
    if ctx is None or not _trace_enabled():
        return
    _LOG.info(
        "mastering_trace_chain %s",
        _fmt_kv(job_id=ctx.job_id, path=ctx.path, filename=ctx.filename, modules=",".join(module_ids)),
    )


def trace_validate_failure(
    ctx: Optional[TraceContext],
    mastered: np.ndarray,
    reason: str,
    sr: int = 44100,
) -> None:
    if ctx is None or not _trace_enabled():
        return
    m = signal_metrics(mastered, sr)
    _LOG.info(
        "mastering_trace_validate_fail %s",
        _fmt_kv(job_id=ctx.job_id, path=ctx.path, filename=ctx.filename, reason=reason[:200], **m),
    )


__all__ = [
    "TraceContext",
    "signal_metrics",
    "trace_stage",
    "trace_job_start",
    "trace_job_done",
    "trace_job_error",
    "trace_chain_modules",
    "trace_validate_failure",
]
