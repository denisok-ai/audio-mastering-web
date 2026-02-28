"""Тесты для ai.py: лимиты, правило-движок рекомендаций, отчёт, nl_to_config.

Без вызова реального LLM API (при отсутствии ключа используются rule-based функции).
Запуск: cd backend && python -m pytest tests/test_ai.py -v
"""
import pytest


def test_get_ai_limit_for_tier():
    from app.ai import get_ai_limit_for_tier
    assert get_ai_limit_for_tier("free") == 5
    assert get_ai_limit_for_tier("pro") == 50
    assert get_ai_limit_for_tier("studio") == -1
    assert get_ai_limit_for_tier("") == 5
    assert get_ai_limit_for_tier(None) == 5


def test_check_ai_rate_limit_fresh_identifier():
    from app.ai import check_ai_rate_limit, _ai_rate_limits
    _ai_rate_limits.clear()
    r = check_ai_rate_limit("user-123", "free")
    assert r["ok"] is True
    assert r["used"] == 0
    assert r["limit"] == 5
    assert r["remaining"] == 5
    assert "reset_at" in r


def test_check_ai_rate_limit_after_record():
    from app.ai import check_ai_rate_limit, record_ai_usage, _ai_rate_limits
    _ai_rate_limits.clear()
    ident = "test-user-456"
    r0 = check_ai_rate_limit(ident, "free")
    assert r0["remaining"] == 5
    record_ai_usage(ident)
    record_ai_usage(ident)
    r1 = check_ai_rate_limit(ident, "free")
    assert r1["used"] == 2
    assert r1["remaining"] == 3
    assert r1["ok"] is True
    _ai_rate_limits.clear()


def test_check_ai_rate_limit_studio_unlimited():
    from app.ai import check_ai_rate_limit, _ai_rate_limits
    _ai_rate_limits.clear()
    r = check_ai_rate_limit("studio-user", "studio")
    assert r["limit"] == -1
    assert r["remaining"] == -1
    assert r["ok"] is True


def test_rule_based_recommend_returns_style_and_lufs():
    from app.ai import recommend_preset
    analysis = {"lufs": -22.0, "duration_sec": 180, "channels": 2}
    out = recommend_preset(analysis)
    assert "style" in out
    assert "target_lufs" in out
    assert "reason" in out
    assert out["style"] in ("standard", "edm", "hiphop", "classical", "podcast", "lofi", "house_basic")
    assert -24 <= out["target_lufs"] <= -6


def test_rule_based_recommend_quiet_with_lows():
    from app.ai import recommend_preset
    analysis = {"lufs": -21.0, "duration_sec": 120, "channels": 2, "spectrum_bars": [0.5] * 16 + [0.1] * 48}
    out = recommend_preset(analysis)
    assert out["style"] in ("edm", "standard", "house_basic")
    assert isinstance(out["target_lufs"], (int, float))


def test_rule_based_recommend_long_mono_podcast():
    from app.ai import recommend_preset
    analysis = {"lufs": -18.0, "duration_sec": 700, "channels": 1}
    out = recommend_preset(analysis)
    assert out["style"] == "podcast"
    assert out["target_lufs"] == -16.0


def test_report_with_recommendations_returns_summary_and_list():
    from app.ai import report_with_recommendations
    analysis = {"lufs": -16.5, "peak_dbfs": -3.0, "duration_sec": 200, "channels": 2}
    out = report_with_recommendations(analysis)
    assert "summary" in out
    assert "recommendations" in out
    assert isinstance(out["recommendations"], list)
    assert len(out["recommendations"]) >= 1
    assert isinstance(out["summary"], str)


def test_report_empty_analysis():
    from app.ai import report_with_recommendations
    out = report_with_recommendations({})
    assert "summary" in out
    assert "recommendations" in out
    assert len(out["recommendations"]) >= 1


def test_nl_to_config_empty_string_returns_unchanged():
    from app.ai import nl_to_config
    current = {"denoise_strength": 0.5}
    out = nl_to_config("", current_config=current)
    assert out["chain_config"] == current
    assert out["target_lufs"] is None


def test_nl_to_config_none_language_returns_unchanged():
    from app.ai import nl_to_config
    out = nl_to_config(None, current_config={})
    assert out["chain_config"] == {}
    assert out["target_lufs"] is None


def test_nl_to_config_without_api_key_returns_unchanged():
    """Без API ключа nl_to_config не вызывает LLM и возвращает текущий конфиг."""
    from app.ai import nl_to_config
    current = {"parallel_mix": 0.3}
    out = nl_to_config("сделай громче", current_config=current)
    assert out["chain_config"] == current
    assert out["target_lufs"] is None


def test_valid_styles_contains_all_preset_names():
    from app.ai import VALID_STYLES
    from app.pipeline import STYLE_CONFIGS
    for style in STYLE_CONFIGS:
        assert style in VALID_STYLES
