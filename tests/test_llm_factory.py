"""Tests for LLM factory."""

from __future__ import annotations

import pytest

from facultyai.config import LLMConfig
from facultyai.llm_factory import get_llm


def test_get_llm_openai() -> None:
    cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
    llm = get_llm(cfg)
    assert llm.model_name == "gpt-4o"


def test_get_llm_deepseek() -> None:
    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
    )
    llm = get_llm(cfg)
    assert llm.model_name == "deepseek-chat"


def test_get_llm_openai_compatible() -> None:
    cfg = LLMConfig(provider="openai_compatible", model="local-model", api_key="sk-fake")
    llm = get_llm(cfg)
    assert llm.model_name == "local-model"


def test_get_llm_unsupported_provider() -> None:
    cfg = LLMConfig(provider="unsupported")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm(cfg)
