"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from facultyai.config import AppConfig, LLMConfig, load_config, mask_secrets


class TestAppConfig:
    def test_default_config(self) -> None:
        cfg = AppConfig()
        assert cfg.version == 2
        assert cfg.llm.provider == "openai_compatible"
        assert cfg.scraping.headless is True
        assert cfg.output.unique_keys == ["Email", "English Full Name"]

    def test_invalid_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(version=1)

    def test_env_var_resolution(
        self, sample_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-1234567890")
        cfg = load_config(sample_config_file)
        assert cfg.llm.api_key == "sk-test-1234567890"
        assert cfg.llm.provider == "deepseek"
        assert cfg.scraping.headless is False
        assert cfg.scraping.max_concurrent_jobs == 5

    def test_missing_env_var_becomes_empty(
        self, sample_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        cfg = load_config(sample_config_file)
        assert cfg.llm.api_key == ""

    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.version == 2


class TestMaskSecrets:
    def test_mask_api_key(self, monkeypatch: pytest.MonkeyPatch, sample_config_file: Path) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmno")
        cfg = load_config(sample_config_file)
        masked = mask_secrets(cfg)
        key = masked["llm"]["api_key"]
        assert "abcdefghijklmno" not in key
        assert key.startswith("sk")
        assert key.endswith("no")
        assert "*" in key

    def test_short_key_fully_masked(self) -> None:
        cfg = AppConfig(llm=LLMConfig(api_key="ab"))
        masked = mask_secrets(cfg)
        assert masked["llm"]["api_key"] == "****"
