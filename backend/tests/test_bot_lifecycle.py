"""Тесты меню команд и списка команд lifecycle бота."""
from app.bot.lifecycle import _admin_bot_commands, _regular_user_bot_commands


def test_regular_user_commands_no_admin():
    cmds = _regular_user_bot_commands()
    names = {c.command for c in cmds}
    assert "admin" not in names
    assert "server" not in names
    assert "broadcast" not in names
    assert "start" in names
    assert "master" in names
    assert "ask" in names
    assert len(cmds) == 12


def test_admin_commands_superset():
    cmds = _admin_bot_commands()
    names = {c.command for c in cmds}
    assert "admin" in names
    assert "server" in names
    assert "stats" in names
    assert "jobs" in names
    assert "errors" in names
    assert "report" in names
    assert "broadcast" in names
    assert "start" in names
    assert len(cmds) == 19
