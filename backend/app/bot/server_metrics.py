"""Метрики сервера для админ-панели Telegram (отчёты на русском)."""
from __future__ import annotations

import os
import shutil
import time
from typing import Any, Optional

from ..config import settings


def _human_gb(n_bytes: float) -> float:
    return round(n_bytes / (1024**3), 2)


def _human_mb(n_bytes: float) -> float:
    return round(n_bytes / (1024**2), 1)


def get_system_metrics() -> dict[str, Any]:
    """
    Собирает метрики CPU, RAM, диск, сеть, uptime.
    Предпочтительно psutil; иначе упрощённый fallback через /proc (Linux).
    """
    out: dict[str, Any] = {
        "ncpu": max(1, os.cpu_count() or 1),
        "loadavg": "—",
        "cpu_percent": None,
        "ram_total_gb": 0.0,
        "ram_used_gb": 0.0,
        "ram_percent": 0.0,
        "swap_total_gb": 0.0,
        "swap_used_gb": 0.0,
        "swap_percent": 0.0,
        "disk_root": {},
        "disk_temp": None,
        "uptime_sec": 0.0,
        "net_sent_mb": 0.0,
        "net_recv_mb": 0.0,
        "process_rss_mb": 0.0,
        "process_cpu_percent": None,
        "source": "fallback",
    }

    try:
        import psutil  # type: ignore

        out["source"] = "psutil"
        try:
            la = os.getloadavg()
            out["loadavg"] = f"{la[0]:.2f} {la[1]:.2f} {la[2]:.2f}"
        except (AttributeError, OSError):
            pass
        out["cpu_percent"] = round(psutil.cpu_percent(interval=0.25), 1)
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = _human_gb(vm.total)
        out["ram_used_gb"] = _human_gb(vm.used)
        out["ram_percent"] = round(vm.percent, 1)
        sw = psutil.swap_memory()
        if sw.total > 0:
            out["swap_total_gb"] = _human_gb(sw.total)
            out["swap_used_gb"] = _human_gb(sw.used)
            out["swap_percent"] = round(sw.percent, 1)
        du_root = shutil.disk_usage("/")
        out["disk_root"] = {
            "total_gb": _human_gb(du_root.total),
            "used_percent": round(100.0 * du_root.used / du_root.total, 1) if du_root.total else 0.0,
            "free_gb": _human_gb(du_root.free),
        }
        temp_dir = (getattr(settings, "temp_dir", "") or "/tmp/masterflow").strip()
        try:
            du_t = shutil.disk_usage(temp_dir)
            out["disk_temp"] = {
                "path": temp_dir[:60],
                "used_percent": round(100.0 * du_t.used / du_t.total, 1) if du_t.total else 0.0,
                "free_gb": _human_gb(du_t.free),
            }
        except OSError:
            pass
        out["uptime_sec"] = float(time.time() - psutil.boot_time())
        try:
            net = psutil.net_io_counters()
            if net:
                out["net_sent_mb"] = round(net.bytes_sent / (1024**2), 1)
                out["net_recv_mb"] = round(net.bytes_recv / (1024**2), 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            p = psutil.Process(os.getpid())
            out["process_rss_mb"] = round(p.memory_info().rss / (1024**2), 1)
            out["process_cpu_percent"] = round(p.cpu_percent(interval=0.1), 1)
        except Exception:  # noqa: BLE001
            pass
        return out
    except ImportError:
        pass

    # Fallback: Linux /proc
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            parts = f.read().split()
            if len(parts) >= 3:
                out["loadavg"] = f"{parts[0]} {parts[1]} {parts[2]}"
    except OSError:
        pass
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, rest = line.split(":", 1)
                    meminfo[k.strip()] = int(rest.split()[0]) * 1024  # kB -> bytes
        total = meminfo.get("MemTotal", 0)
        avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        if total > 0:
            used = total - avail
            out["ram_total_gb"] = _human_gb(total)
            out["ram_used_gb"] = _human_gb(used)
            out["ram_percent"] = round(100.0 * used / total, 1)
    except OSError:
        pass
    du_root = shutil.disk_usage("/")
    out["disk_root"] = {
        "total_gb": _human_gb(du_root.total),
        "used_percent": round(100.0 * du_root.used / du_root.total, 1) if du_root.total else 0.0,
        "free_gb": _human_gb(du_root.free),
    }
    temp_dir = (getattr(settings, "temp_dir", "") or "/tmp/masterflow").strip()
    try:
        du_t = shutil.disk_usage(temp_dir)
        out["disk_temp"] = {
            "path": temp_dir[:60],
            "used_percent": round(100.0 * du_t.used / du_t.total, 1) if du_t.total else 0.0,
            "free_gb": _human_gb(du_t.free),
        }
    except OSError:
        pass
    try:
        with open("/proc/uptime", encoding="utf-8") as f:
            out["uptime_sec"] = float(f.read().split()[0])
    except OSError:
        pass
    approx_cpu = None
    try:
        la0 = float(out["loadavg"].split()[0])
        approx_cpu = min(100.0, round(100.0 * la0 / out["ncpu"], 1))
    except (ValueError, IndexError):
        pass
    out["cpu_percent"] = approx_cpu
    return out


def format_server_report(m: Optional[dict] = None) -> str:
    """Текст отчёта о сервере для Telegram (HTML)."""
    m = m or get_system_metrics()
    dr = m.get("disk_root") or {}
    dt = m.get("disk_temp")
    cpu = m.get("cpu_percent")
    cpu_s = f"{cpu}%" if cpu is not None else "н/д"
    pcpu = m.get("process_cpu_percent")
    pcpu_s = f"{pcpu}%" if pcpu is not None else "н/д"
    up_h = round(m.get("uptime_sec", 0) / 3600.0, 1)
    lines = [
        "🖥 <b>Сервер</b>",
        f"Ядер CPU: <b>{m.get('ncpu', 1)}</b>",
        f"Средняя нагрузка (1/5/15 мин): <code>{m.get('loadavg', '—')}</code>",
        f"Загрузка CPU (оценка): <b>{cpu_s}</b>",
        "",
        "💾 <b>Память</b>",
        f"RAM: <b>{m.get('ram_used_gb', 0)}</b> / {m.get('ram_total_gb', 0)} ГБ "
        f"({m.get('ram_percent', 0)}%)",
    ]
    if m.get("swap_total_gb", 0) > 0:
        lines.append(
            f"Swap: {m.get('swap_used_gb', 0)} / {m.get('swap_total_gb', 0)} ГБ "
            f"({m.get('swap_percent', 0)}%)"
        )
    lines.extend(
        [
            "",
            "📀 <b>Диск</b>",
            f"Корень /: занято <b>{dr.get('used_percent', 0)}%</b>, "
            f"свободно {dr.get('free_gb', 0)} ГБ из {dr.get('total_gb', 0)} ГБ",
        ]
    )
    if dt:
        lines.append(
            f"Каталог temp (<code>{dt.get('path', '')}</code>): "
            f"занято <b>{dt.get('used_percent', 0)}%</b>, свободно {dt.get('free_gb', 0)} ГБ"
        )
    lines.extend(
        [
            "",
            "⚙️ <b>Процесс приложения</b>",
            f"RSS: <b>{m.get('process_rss_mb', 0)}</b> МБ, CPU: {pcpu_s}",
            "",
            "🌐 <b>Сеть (с момента загрузки ОС)</b>",
            f"Отправлено: {m.get('net_sent_mb', 0)} МБ, получено: {m.get('net_recv_mb', 0)} МБ",
            "",
            f"⏱ Аптайм ОС: <b>{up_h}</b> ч",
            f"<i>Источник метрик: {m.get('source', '?')}</i>",
        ]
    )
    return "\n".join(lines)


def format_server_oneline(m: Optional[dict] = None) -> str:
    """Краткая строка для /server."""
    m = m or get_system_metrics()
    dr = m.get("disk_root") or {}
    cpu = m.get("cpu_percent")
    cpu_s = f"{cpu}%" if cpu is not None else "н/д"
    return (
        f"🖥 CPU ~{cpu_s}, RAM {m.get('ram_percent', 0)}%, "
        f"диск / {dr.get('used_percent', 0)}%, "
        f"нагрузка <code>{m.get('loadavg', '—')}</code>"
    )
