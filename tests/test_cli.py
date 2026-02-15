"""CLI(cli.py) 모듈의 통합 테스트입니다.

typer.testing.CliRunner를 사용하여 각 명령어의 동작을 검증합니다.
외부 파이프라인 호출은 mock으로 대체합니다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from slm_factory.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionCommand:
    """version 명령어의 테스트입니다."""

    def test_버전_출력(self):
        """출력에 '0.1.0'이 포함되는지 확인합니다."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class TestInitCommand:
    """init 명령어의 테스트입니다."""

    def test_프로젝트_디렉토리_생성(self, tmp_path):
        """프로젝트 디렉토리, documents, output, project.yaml이 생성되는지 확인합니다."""
        result = runner.invoke(app, [
            "init",
            "--name", "test-proj",
            "--path", str(tmp_path),
        ])

        assert result.exit_code == 0

        project_dir = tmp_path / "test-proj"
        assert project_dir.is_dir()
        assert (project_dir / "documents").is_dir()
        assert (project_dir / "output").is_dir()
        assert (project_dir / "project.yaml").is_file()

        # project.yaml에 프로젝트명이 포함되는지 확인
        content = (project_dir / "project.yaml").read_text(encoding="utf-8")
        assert "test-proj" in content


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRunCommand:
    """run 명령어의 테스트입니다."""

    def test_존재하지_않는_config_exit_code_1(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["run", "--config", "/nonexistent/path.yaml"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


class TestParseCommand:
    """parse 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        """_load_pipeline이 올바르게 호출되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()

        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(app, ["parse", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_pipeline.step_parse.assert_called_once()


# ---------------------------------------------------------------------------
# no args
# ---------------------------------------------------------------------------


class TestNoArgs:
    """인자 없이 실행했을 때의 테스트입니다."""

    def test_도움말_출력(self):
        """인자 없이 실행하면 도움말 메시지가 출력되는지 확인합니다."""
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    """generate 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        """_load_pipeline이 올바르게 호출되고 파싱+생성이 수행되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()

        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(app, ["generate", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_pipeline.step_parse.assert_called_once()
        mock_pipeline.step_generate.assert_called_once()


# ---------------------------------------------------------------------------
# wizard
# ---------------------------------------------------------------------------


class TestWizardCommand:
    """wizard 명령어의 테스트입니다."""

    def test_존재하지_않는_config_exit_code_1(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["wizard", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1

    def test_wizard_help_포함(self):
        """wizard 명령어의 도움말에 '대화형'이 포함되는지 확인합니다."""
        result = runner.invoke(app, ["wizard", "--help"])
        assert result.exit_code == 0
        assert "대화형" in result.output
