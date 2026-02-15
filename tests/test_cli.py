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


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """config.py 필드 검증 테스트입니다."""

    def test_프로젝트명_빈문자열_거부(self):
        """ProjectConfig.name에 빈 문자열을 넣으면 ValidationError가 발생하는지 확인합니다."""
        from pydantic import ValidationError
        from slm_factory.config import ProjectConfig
        with pytest.raises(ValidationError):
            ProjectConfig(name="")

    def test_교사모델명_빈문자열_거부(self):
        """TeacherConfig.model에 빈 문자열을 넣으면 ValidationError가 발생하는지 확인합니다."""
        from pydantic import ValidationError
        from slm_factory.config import TeacherConfig
        with pytest.raises(ValidationError):
            TeacherConfig(model="")

    def test_교사_api_base_빈문자열_거부(self):
        from pydantic import ValidationError
        from slm_factory.config import TeacherConfig
        with pytest.raises(ValidationError):
            TeacherConfig(api_base="")

    def test_학생모델명_빈문자열_거부(self):
        from pydantic import ValidationError
        from slm_factory.config import StudentConfig
        with pytest.raises(ValidationError):
            StudentConfig(model="")

    def test_ollama_모델명_빈문자열_거부(self):
        from pydantic import ValidationError
        from slm_factory.config import OllamaExportConfig
        with pytest.raises(ValidationError):
            OllamaExportConfig(model_name="")

    def test_정상값_통과(self):
        """기본값이 유효한지 확인합니다."""
        from slm_factory.config import ProjectConfig, TeacherConfig, StudentConfig, OllamaExportConfig
        # 기본값으로 생성 시 에러 없어야 함
        ProjectConfig()
        TeacherConfig()
        StudentConfig()
        OllamaExportConfig()


# ---------------------------------------------------------------------------
# wizard --resume
# ---------------------------------------------------------------------------


class TestWizardResume:
    """wizard --resume 옵션 테스트입니다."""

    def test_resume_옵션_존재(self):
        """wizard --help에 --resume 옵션이 표시되는지 확인합니다."""
        result = runner.invoke(app, ["wizard", "--help"])
        assert result.exit_code == 0
        assert "--resume" in result.output


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


class TestCheckCommand:
    """check 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["check", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """status 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["status", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


class TestCleanCommand:
    """clean 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, ["clean", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


class TestScoreCommand:
    """score 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.step_validate.return_value = [MagicMock()]
        mock_pipeline.step_score.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(app, ["score", "--config", "test.yaml"])
        assert result.exit_code == 0
        mock_pipeline.step_score.assert_called_once()


# ---------------------------------------------------------------------------
# augment
# ---------------------------------------------------------------------------


class TestAugmentCommand:
    """augment 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.step_validate.return_value = [MagicMock()]
        mock_pipeline.step_score.return_value = [MagicMock()]
        mock_pipeline.step_augment.return_value = [MagicMock()]
        mock_pipeline.config.paths.ensure_dirs = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(app, ["augment", "--config", "test.yaml"])
        assert result.exit_code == 0
        mock_pipeline.step_augment.assert_called_once()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    """analyze 명령어의 테스트입니다."""

    def test_load_pipeline_호출(self, mocker):
        mock_pipeline = MagicMock()
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline.step_validate.return_value = [MagicMock()]
        mock_pipeline.step_score.return_value = [MagicMock()]
        mock_pipeline.step_augment.return_value = [MagicMock()]
        mock_pipeline.step_analyze.return_value = None
        mock_pipeline.config.paths.ensure_dirs = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(app, ["analyze", "--config", "test.yaml"])
        assert result.exit_code == 0
        mock_pipeline.step_analyze.assert_called_once()


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


class TestConvertCommand:
    """convert 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        result = runner.invoke(app, ["convert", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExportCommand:
    """export 명령어의 테스트입니다."""

    def test_존재하지_않는_config(self):
        result = runner.invoke(app, ["export", "--config", "/nonexistent/path.yaml"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# _load_qa_data helper
# ---------------------------------------------------------------------------


class TestLoadQaData:
    """_load_qa_data 헬퍼 함수의 테스트입니다."""

    def test_명시적_경로_파일_존재(self, tmp_path):
        """--data 옵션으로 지정한 파일이 존재하면 해당 파일을 로드하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        data_file = tmp_path / "test.json"
        data_file.write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline._load_pairs.return_value = [mock_pair]

        result = _load_qa_data(pipeline, str(data_file))

        pipeline._load_pairs.assert_called_once_with(data_file)
        assert result == [mock_pair]

    def test_명시적_경로_파일_미존재(self):
        """--data 옵션으로 지정한 파일이 없으면 Exit 예외가 발생하는지 확인합니다."""
        from click.exceptions import Exit as ClickExit

        from slm_factory.cli import _load_qa_data

        pipeline = MagicMock()

        with pytest.raises(ClickExit):
            _load_qa_data(pipeline, "/nonexistent/file.json")

    def test_자동감지_qa_augmented(self, tmp_path):
        """출력 디렉토리에서 qa_augmented.json을 자동 감지하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        qa_file = tmp_path / "qa_augmented.json"
        qa_file.write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._load_pairs.return_value = [mock_pair]

        result = _load_qa_data(pipeline, None)

        pipeline._load_pairs.assert_called_once_with(qa_file)
        assert result == [mock_pair]

    def test_자동감지_우선순위(self, tmp_path):
        """qa_augmented > qa_scored > qa_alpaca 우선순위로 감지하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        (tmp_path / "qa_augmented.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_scored.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_alpaca.json").write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._load_pairs.return_value = [mock_pair]

        _load_qa_data(pipeline, None)

        pipeline._load_pairs.assert_called_once_with(tmp_path / "qa_augmented.json")

    def test_extra_candidates_우선(self, tmp_path):
        """extra_candidates가 기본 후보보다 우선하는지 확인합니다."""
        from slm_factory.cli import _load_qa_data

        (tmp_path / "qa_reviewed.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_augmented.json").write_text("[]", encoding="utf-8")

        mock_pair = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._load_pairs.return_value = [mock_pair]

        _load_qa_data(pipeline, None, extra_candidates=["qa_reviewed.json"])

        pipeline._load_pairs.assert_called_once_with(tmp_path / "qa_reviewed.json")

    def test_파일_미발견_exit(self, tmp_path):
        """출력 디렉토리에 QA 파일이 없으면 Exit 예외가 발생하는지 확인합니다."""
        from click.exceptions import Exit as ClickExit

        from slm_factory.cli import _load_qa_data

        pipeline = MagicMock()
        pipeline.output_dir = tmp_path

        with pytest.raises(ClickExit):
            _load_qa_data(pipeline, None)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


class TestEvalCommand:
    """eval 명령어의 테스트입니다."""

    def test_모델_평가_실행(self, mocker, tmp_path):
        """파이프라인과 평가기를 올바르게 호출하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.config.eval.output_file = "eval_results.json"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = [MagicMock()]
        mocker.patch(
            "slm_factory.evaluator.ModelEvaluator",
            return_value=mock_evaluator,
        )

        result = runner.invoke(app, [
            "eval", "--config", "test.yaml", "--model", "test-model",
        ])

        assert result.exit_code == 0
        mock_evaluator.evaluate.assert_called_once()
        mock_evaluator.save_results.assert_called_once()
        mock_evaluator.print_summary.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "eval", "--config", "/nonexistent/path.yaml", "--model", "test",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# export-gguf
# ---------------------------------------------------------------------------


class TestExportGgufCommand:
    """export-gguf 명령어의 테스트입니다."""

    def test_gguf_변환_실행(self, mocker, tmp_path):
        """파이프라인과 GGUFExporter를 올바르게 호출하는지 확인합니다."""
        model_dir = tmp_path / "merged_model"
        model_dir.mkdir()

        mock_pipeline = MagicMock()
        mock_pipeline.config.paths.output = tmp_path
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = tmp_path / "model.gguf"
        mocker.patch(
            "slm_factory.exporter.gguf_export.GGUFExporter",
            return_value=mock_exporter,
        )

        result = runner.invoke(app, [
            "export-gguf", "--config", "test.yaml",
            "--model-dir", str(model_dir),
        ])

        assert result.exit_code == 0
        mock_exporter.export.assert_called_once()

    def test_모델_디렉토리_미존재(self, mocker, tmp_path):
        """모델 디렉토리가 존재하지 않으면 exit code 1로 종료하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.config.paths.output = tmp_path / "nonexistent"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        result = runner.invoke(app, [
            "export-gguf", "--config", "test.yaml",
            "--model-dir", "/nonexistent/dir",
        ])

        assert result.exit_code == 1

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "export-gguf", "--config", "/nonexistent/path.yaml",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdateCommand:
    """update 명령어의 테스트입니다."""

    def test_증분_업데이트_실행(self, mocker, tmp_path):
        """변경 문서 감지 및 증분 업데이트가 수행되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.step_parse.return_value = [MagicMock()]
        mock_pipeline.step_generate.return_value = [MagicMock()]
        mock_pipeline._load_pairs.return_value = []
        mock_pipeline.config.incremental.merge_strategy = "append"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        mock_tracker = MagicMock()
        mock_tracker.get_changed_files.return_value = [Path("doc1.pdf")]
        mock_tracker.merge_qa_pairs.return_value = [MagicMock()]
        mocker.patch(
            "slm_factory.incremental.IncrementalTracker",
            return_value=mock_tracker,
        )

        result = runner.invoke(app, ["update", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_tracker.get_changed_files.assert_called_once()
        mock_pipeline.step_parse.assert_called_once()
        mock_pipeline.step_generate.assert_called_once()

    def test_변경_없음_조기종료(self, mocker):
        """변경된 문서가 없으면 조기 종료하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)

        mock_tracker = MagicMock()
        mock_tracker.get_changed_files.return_value = []
        mocker.patch(
            "slm_factory.incremental.IncrementalTracker",
            return_value=mock_tracker,
        )

        result = runner.invoke(app, ["update", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_pipeline.step_parse.assert_not_called()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "update", "--config", "/nonexistent/path.yaml",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# generate-dialogue
# ---------------------------------------------------------------------------


class TestGenerateDialogueCommand:
    """generate-dialogue 명령어의 테스트입니다."""

    def test_대화_생성_실행(self, mocker, tmp_path):
        """파이프라인과 DialogueGenerator를 올바르게 호출하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_teacher = MagicMock()
        mocker.patch(
            "slm_factory.teacher.create_teacher", return_value=mock_teacher,
        )

        mock_generator = MagicMock()
        mocker.patch(
            "slm_factory.teacher.dialogue_generator.DialogueGenerator",
            return_value=mock_generator,
        )

        mocker.patch("asyncio.run", return_value=[MagicMock()])

        result = runner.invoke(app, [
            "generate-dialogue", "--config", "test.yaml",
        ])

        assert result.exit_code == 0
        mock_generator.save_dialogues.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "generate-dialogue", "--config", "/nonexistent/path.yaml",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompareCommand:
    """compare 명령어의 테스트입니다."""

    def test_모델_비교_실행(self, mocker, tmp_path):
        """파이프라인과 ModelComparator를 올바르게 호출하는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.config.compare.output_file = "compare_results.json"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_comparator = MagicMock()
        mock_comparator.compare.return_value = [MagicMock()]
        mocker.patch(
            "slm_factory.comparator.ModelComparator",
            return_value=mock_comparator,
        )

        result = runner.invoke(app, [
            "compare", "--config", "test.yaml",
            "--base-model", "base", "--finetuned-model", "finetuned",
        ])

        assert result.exit_code == 0
        mock_comparator.compare.assert_called_once()
        mock_comparator.save_results.assert_called_once()
        mock_comparator.print_summary.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "compare", "--config", "/nonexistent/path.yaml",
            "--base-model", "base", "--finetuned-model", "finetuned",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------


class TestDashboardCommand:
    """dashboard 명령어의 테스트입니다."""

    def test_대시보드_실행(self, mocker, tmp_path):
        """load_config과 PipelineDashboard가 올바르게 호출되는지 확인합니다."""
        mock_cfg = MagicMock()
        mock_cfg.paths.output = str(tmp_path / "output")
        mock_cfg.dashboard.refresh_interval = 5
        mocker.patch("slm_factory.cli._find_config", return_value="test.yaml")
        mocker.patch("slm_factory.config.load_config", return_value=mock_cfg)

        mock_dashboard = MagicMock()
        mocker.patch(
            "slm_factory.tui.dashboard.PipelineDashboard",
            return_value=mock_dashboard,
        )

        result = runner.invoke(app, ["dashboard", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_dashboard.run.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "dashboard", "--config", "/nonexistent/path.yaml",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


class TestReviewCommand:
    """review 명령어의 테스트입니다."""

    def test_리뷰_실행(self, mocker, tmp_path):
        """파이프라인과 QAReviewerApp이 올바르게 호출되는지 확인합니다."""
        mock_pipeline = MagicMock()
        mock_pipeline.output_dir = tmp_path / "output"
        mock_pipeline.config.review.output_file = "qa_reviewed.json"
        mocker.patch("slm_factory.cli._load_pipeline", return_value=mock_pipeline)
        mocker.patch("slm_factory.cli._load_qa_data", return_value=[MagicMock()])

        mock_reviewer_cls = MagicMock()
        mock_reviewer_cls.count_statuses.return_value = {
            "approved": 1, "rejected": 0, "pending": 0,
        }
        mocker.patch(
            "slm_factory.tui.reviewer.QAReviewerApp", mock_reviewer_cls,
        )

        result = runner.invoke(app, ["review", "--config", "test.yaml"])

        assert result.exit_code == 0
        mock_reviewer_cls.return_value.run.assert_called_once()
        mock_reviewer_cls.count_statuses.assert_called_once()

    def test_존재하지_않는_config(self):
        """존재하지 않는 설정 파일을 지정하면 exit code 1로 종료하는지 확인합니다."""
        result = runner.invoke(app, [
            "review", "--config", "/nonexistent/path.yaml",
        ])
        assert result.exit_code == 1
