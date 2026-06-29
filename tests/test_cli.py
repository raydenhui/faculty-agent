"""Tests for the CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from facultyai.cli import cli


class TestCLI:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "resume" in result.output
        assert "export" in result.output
        assert "chat" in result.output

    def test_config_validate(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate", "--path", "config.yaml"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_config_show_masks_secrets(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key-123456")
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--path", "config.yaml"])
        assert result.exit_code == 0
        assert "secret-key-123456" not in result.output
