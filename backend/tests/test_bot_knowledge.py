"""Tests for bot knowledge base and AI consultant integration."""
import pytest


def test_get_product_knowledge_ru():
    from app.bot.knowledge_base import get_product_knowledge
    kb = get_product_knowledge("ru")
    assert isinstance(kb, str)
    assert len(kb) > 500
    assert "Magic Master" in kb
    assert "standard" in kb
    assert "edm" in kb
    assert "LUFS" in kb
    assert "Free" in kb
    assert "Pro" in kb
    assert "Studio" in kb
    assert "/master" in kb


def test_get_product_knowledge_en():
    from app.bot.knowledge_base import get_product_knowledge
    kb = get_product_knowledge("en")
    assert isinstance(kb, str)
    assert len(kb) > 500
    assert "Magic Master" in kb
    assert "mastering" in kb.lower()
    assert "Upload" in kb or "upload" in kb
    assert "Free" in kb
    assert "Pro" in kb


def test_get_product_knowledge_defaults_to_ru():
    from app.bot.knowledge_base import get_product_knowledge
    kb_ru = get_product_knowledge("ru")
    kb_default = get_product_knowledge("")
    assert kb_default == kb_ru
    kb_none = get_product_knowledge(None)
    assert kb_none == kb_ru


def test_knowledge_ru_and_en_are_different():
    from app.bot.knowledge_base import get_product_knowledge
    kb_ru = get_product_knowledge("ru")
    kb_en = get_product_knowledge("en")
    assert kb_ru != kb_en


def test_knowledge_contains_all_styles():
    from app.bot.knowledge_base import get_product_knowledge
    from app.pipeline import STYLE_CONFIGS
    kb_ru = get_product_knowledge("ru")
    kb_en = get_product_knowledge("en")
    for style in STYLE_CONFIGS:
        assert style in kb_ru, f"Style '{style}' missing from RU knowledge base"
        assert style in kb_en, f"Style '{style}' missing from EN knowledge base"


def test_chat_assistant_includes_knowledge_in_prompt():
    """Verify chat_assistant builds system prompt with knowledge base."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test reply"))]
    )

    with patch("app.ai._get_llm_client", return_value=(mock_client, "test-model")):
        from app.ai import chat_assistant
        result = chat_assistant(
            [{"role": "user", "content": "hello"}],
            context={"lang": "en"},
        )

    assert result == "Test reply"
    call_args = mock_client.chat.completions.create.call_args
    system_msg = call_args.kwargs["messages"][0]["content"]
    assert "PRODUCT_KNOWLEDGE" in system_msg
    assert "Mastering Styles" in system_msg or "mastering" in system_msg.lower()


def test_chat_assistant_uses_ru_knowledge_for_ru_lang():
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Ответ"))]
    )

    with patch("app.ai._get_llm_client", return_value=(mock_client, "test-model")):
        from app.ai import chat_assistant
        chat_assistant(
            [{"role": "user", "content": "привет"}],
            context={"lang": "ru"},
        )

    call_args = mock_client.chat.completions.create.call_args
    system_msg = call_args.kwargs["messages"][0]["content"]
    assert "Стили мастеринга" in system_msg or "О продукте" in system_msg
