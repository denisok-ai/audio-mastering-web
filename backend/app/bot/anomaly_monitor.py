"""Фоновая проверка аномалий: алерты админу в Telegram (русский текст)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ..config import settings
from ..database import DB_AVAILABLE, SessionLocal
from .. import jobs_store as js
from ..notifier import notify_alert_queue_threshold, notify_operational_anomaly
from .admin_reports import mastering_error_rate_last_hour
from .server_metrics import get_system_metrics

logger = logging.getLogger(__name__)

_task: Optional[asyncio.Task] = None
_stop = asyncio.Event()


def _cfg_int(name: str, default: int) -> int:
    v = getattr(settings, name, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _cfg_float(name: str, default: float) -> float:
    v = getattr(settings, name, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _run_checks_sync() -> None:
    if not getattr(settings, "alert_monitoring_enabled", False):
        return
    m = get_system_metrics()
    cpu = m.get("cpu_percent")
    if cpu is not None and cpu >= _cfg_float("anomaly_cpu_threshold", 85.0):
        notify_operational_anomaly(
            "cpu",
            "Высокая загрузка CPU",
            f"Загрузка CPU: <b>{cpu}%</b>\n"
            f"Нагрузка (1/5/15 мин): <code>{m.get('loadavg', '—')}</code>\n"
            f"Ядер: {m.get('ncpu', 1)}",
        )
    ram_p = float(m.get("ram_percent") or 0)
    if ram_p >= _cfg_float("anomaly_ram_threshold", 85.0):
        notify_operational_anomaly(
            "ram",
            "Высокое использование RAM",
            f"RAM: <b>{ram_p}%</b> "
            f"({m.get('ram_used_gb', 0)} / {m.get('ram_total_gb', 0)} ГБ)",
        )
    dr = m.get("disk_root") or {}
    used = float(dr.get("used_percent") or 0)
    if used >= _cfg_float("anomaly_disk_threshold", 85.0):
        notify_operational_anomaly(
            "disk",
            "Мало свободного места на диске",
            f"Корень /: занято <b>{used}%</b>, свободно {dr.get('free_gb', '—')} ГБ",
        )
    rss = float(m.get("process_rss_mb") or 0)
    rss_max = _cfg_int("anomaly_rss_max_mb", 8192)
    if rss_max > 0 and rss >= rss_max:
        notify_operational_anomaly(
            "rss",
            "Большой объём памяти процесса приложения",
            f"RSS процесса: <b>{rss}</b> МБ (порог {rss_max} МБ)",
        )
    jobs = js.all_jobs()
    run = sum(1 for j in jobs.values() if j.get("status") == "running")
    total = len(jobs)
    thr = int(getattr(settings, "alert_queue_threshold", 0) or 0)
    if thr > 0:
        notify_alert_queue_threshold(total, run)
    min_j = _cfg_int("anomaly_min_jobs_for_error_rate", 8)
    err_thr = _cfg_float("anomaly_error_rate_threshold", 35.0)
    if DB_AVAILABLE and SessionLocal is not None and min_j > 0:
        db = SessionLocal()
        try:
            pair = mastering_error_rate_last_hour(db)
            if pair:
                err_n, tot = pair
                if tot >= min_j and tot > 0:
                    rate = 100.0 * err_n / tot
                    if rate >= err_thr:
                        notify_operational_anomaly(
                            "err_rate",
                            "Много ошибок мастеринга за час",
                            f"Ошибок: <b>{err_n}</b> из <b>{tot}</b> "
                            f"(<b>{rate:.1f}%</b>, порог {err_thr}%)",
                        )
        finally:
            db.close()


async def _loop() -> None:
    interval = max(15, _cfg_int("anomaly_check_interval", 60))
    await asyncio.sleep(5)
    while not _stop.is_set():
        try:
            await asyncio.to_thread(_run_checks_sync)
        except Exception as e:  # noqa: BLE001
            logger.debug("anomaly_monitor: %s", e)
        try:
            await asyncio.wait_for(_stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def anomaly_monitor_start() -> None:
    """Запускает фоновую задачу (идемпотентно)."""
    global _task
    if _task is not None and not _task.done():
        return
    if not getattr(settings, "alert_monitoring_enabled", False):
        logger.info("anomaly_monitor: выключен (alert_monitoring_enabled=0)")
        return
    _stop.clear()
    _task = asyncio.create_task(_loop(), name="anomaly_monitor")
    logger.info("anomaly_monitor: запущен, интервал %s с", getattr(settings, "anomaly_check_interval", 60))


async def anomaly_monitor_shutdown() -> None:
    """Корректное завершение при остановке приложения."""
    global _task
    _stop.set()
    t = _task
    _task = None
    if t and not t.done():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
