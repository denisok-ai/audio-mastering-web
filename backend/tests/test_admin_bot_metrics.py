"""Тесты админ-метрик бота и server_metrics."""
import pytest


def test_format_server_report_russian():
    from app.bot.server_metrics import format_server_report

    m = {
        "ncpu": 4,
        "loadavg": "0.5 0.4 0.3",
        "cpu_percent": 12.5,
        "ram_total_gb": 8.0,
        "ram_used_gb": 4.0,
        "ram_percent": 50.0,
        "swap_total_gb": 0.0,
        "swap_used_gb": 0.0,
        "swap_percent": 0.0,
        "disk_root": {"total_gb": 100.0, "used_percent": 40.0, "free_gb": 60.0},
        "disk_temp": {"path": "/tmp", "used_percent": 10.0, "free_gb": 5.0},
        "uptime_sec": 3600.0,
        "net_sent_mb": 1.0,
        "net_recv_mb": 2.0,
        "process_rss_mb": 200.0,
        "process_cpu_percent": 5.0,
        "source": "test",
    }
    s = format_server_report(m)
    assert "Сервер" in s
    assert "Память" in s
    assert "Диск" in s
    assert "12.5%" in s or "12.5" in s


def test_format_jobs_ru_contains_semaphore():
    from app.bot.admin_reports import format_jobs_ru

    t = format_jobs_ru()
    assert "Задачи мастеринга" in t
    assert "Семафоры" in t


def test_split_telegram_message():
    from app.bot.handlers.admin import _split_telegram_html

    long = "a\n" * 2500
    parts = _split_telegram_html(long, max_len=100)
    assert len(parts) >= 2
    assert all(len(p) <= 100 for p in parts)


def test_list_recent_error_jobs_empty():
    from app.jobs_store import list_recent_error_jobs

    assert isinstance(list_recent_error_jobs(5), list)


def test_mastering_error_rate_none_without_db(monkeypatch):
    from app.bot.admin_reports import mastering_error_rate_last_hour

    assert mastering_error_rate_last_hour(None) is None
