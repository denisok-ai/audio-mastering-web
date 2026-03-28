"""Тесты меню команд и списка команд lifecycle бота."""
from app.bot.lifecycle import _user_bot_commands


def test_user_bot_commands_count_and_names():
    cmds = _user_bot_commands()
    assert len(cmds) == 13
    names = {c.command for c in cmds}
    assert "start" in names
    assert "admin" in names
    assert "master" in names
    assert "ask" in names
