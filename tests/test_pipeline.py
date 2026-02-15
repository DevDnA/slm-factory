"""파이프라인(pipeline.py) 모듈의 통합 테스트입니다.

Pipeline 클래스의 각 단계(parse, generate, validate, convert, train, export)와
전체 실행(run)을 검증합니다. 모든 외부 의존성은 mock으로 대체합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_factory.config import SLMConfig
from slm_factory.pipeline import Pipeline


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_pipeline(make_config, tmp_path) -> Pipeline:
    """테스트용 Pipeline 인스턴스를 생성합니다."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(
        paths={"output": str(output_dir), "documents": str(docs_dir)},
    )
    return Pipeline(config)


# ---------------------------------------------------------------------------
# Pipeline.__init__
# ---------------------------------------------------------------------------


class TestPipelineInit:
    """Pipeline 초기화 테스트입니다."""

    def test_config_및_output_dir_설정(self, make_config, tmp_path):
        """config와 output_dir 필드가 올바르게 설정되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)

        assert isinstance(pipeline.config, SLMConfig)
        assert pipeline.output_dir == Path(str(tmp_path / "output"))


# ---------------------------------------------------------------------------
# step_parse
# ---------------------------------------------------------------------------


class TestStepParse:
    """Pipeline.step_parse 메서드의 테스트입니다."""

    def test_정상_문서_반환(self, make_config, make_parsed_doc, tmp_path, mocker):
        """parse_directory가 문서를 반환할 때 정상적으로 처리되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        mock_doc = make_parsed_doc(doc_id="test.pdf", title="테스트", content="내용입니다.")

        mocker.patch(
            "slm_factory.parsers.registry.parse_directory",
            return_value=[mock_doc],
        )

        docs = pipeline.step_parse()

        assert len(docs) == 1
        assert docs[0].doc_id == "test.pdf"

    def test_문서_없으면_RuntimeError(self, make_config, tmp_path, mocker):
        """파싱된 문서가 없으면 RuntimeError를 발생시키는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)

        mocker.patch(
            "slm_factory.parsers.registry.parse_directory",
            return_value=[],
        )

        with pytest.raises(RuntimeError):
            pipeline.step_parse()

    def test_parsed_documents_json_저장(self, make_config, make_parsed_doc, tmp_path, mocker):
        """파싱 결과가 parsed_documents.json 파일로 저장되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        mock_doc = make_parsed_doc()

        mocker.patch(
            "slm_factory.parsers.registry.parse_directory",
            return_value=[mock_doc],
        )

        pipeline.step_parse()

        saved_file = pipeline.output_dir / "parsed_documents.json"
        assert saved_file.exists()


# ---------------------------------------------------------------------------
# step_generate
# ---------------------------------------------------------------------------


class TestStepGenerate:
    """Pipeline.step_generate 메서드의 테스트입니다."""

    def test_QAPair_반환(self, make_config, make_parsed_doc, make_qa_pair, tmp_path, mocker):
        """QAGenerator를 mock하여 QAPair 리스트를 반환하는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        mock_doc = make_parsed_doc()
        expected_pair = make_qa_pair()

        mock_generator_cls = mocker.patch("slm_factory.teacher.qa_generator.QAGenerator")
        mock_generator = mock_generator_cls.return_value
        mock_generator.generate_all_async = MagicMock(return_value=[expected_pair])

        # asyncio.run을 mock하여 코루틴 실행 없이 결과 반환
        mocker.patch("slm_factory.pipeline.asyncio.run", return_value=[expected_pair])

        pairs = pipeline.step_generate([mock_doc])

        assert len(pairs) == 1
        assert pairs[0].question == expected_pair.question


# ---------------------------------------------------------------------------
# step_validate
# ---------------------------------------------------------------------------


class TestStepValidate:
    """Pipeline.step_validate 메서드의 테스트입니다."""

    def test_validation_비활성화시_모든_쌍_반환(self, make_config, make_qa_pair, tmp_path):
        """validation.enabled=False일 때 입력 쌍을 그대로 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            validation={"enabled": False},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair(), make_qa_pair(question="두 번째 질문")]
        result = pipeline.step_validate(pairs)

        assert len(result) == len(pairs)

    def test_validation_활성화시_RuleValidator_호출(self, make_config, make_qa_pair, tmp_path, mocker):
        """validation.enabled=True일 때 RuleValidator.validate_batch가 호출되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        pairs = [make_qa_pair()]

        mock_validator_cls = mocker.patch("slm_factory.validator.rules.RuleValidator")
        mock_validator = mock_validator_cls.return_value
        mock_validator.validate_batch.return_value = (pairs, [])

        result = pipeline.step_validate(pairs)

        assert len(result) == 1

    def test_groundedness_활성화시_GroundednessChecker_호출(
        self, make_config, make_parsed_doc, make_qa_pair, tmp_path, mocker
    ):
        """groundedness가 활성화되어 있고 docs가 제공되면 GroundednessChecker가 호출되는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            validation={"enabled": True, "groundedness": {"enabled": True, "threshold": 0.5}},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        docs = [make_parsed_doc()]

        mock_rule_cls = mocker.patch("slm_factory.validator.rules.RuleValidator")
        mock_rule_cls.return_value.validate_batch.return_value = (pairs, [])

        mock_ground_cls = mocker.patch("slm_factory.validator.similarity.GroundednessChecker")
        mock_ground = mock_ground_cls.return_value
        mock_ground.check_batch.return_value = (pairs, [])

        result = pipeline.step_validate(pairs, docs=docs)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# step_convert
# ---------------------------------------------------------------------------


class TestStepConvert:
    """Pipeline.step_convert 메서드의 테스트입니다."""

    def test_ChatFormatter_호출(self, make_config, make_qa_pair, tmp_path, mocker):
        """ChatFormatter.save_training_data가 올바르게 호출되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        pairs = [make_qa_pair()]
        expected_path = pipeline.output_dir / "training_data.jsonl"

        mock_formatter_cls = mocker.patch("slm_factory.converter.ChatFormatter")
        mock_formatter = mock_formatter_cls.return_value
        mock_formatter.save_training_data.return_value = expected_path

        result = pipeline.step_convert(pairs)

        assert result == expected_path


# ---------------------------------------------------------------------------
# step_train
# ---------------------------------------------------------------------------


class TestStepTrain:
    """Pipeline.step_train 메서드의 테스트입니다."""

    def test_DataLoader_및_LoRATrainer_호출(self, make_config, tmp_path, mocker):
        """DataLoader와 LoRATrainer가 올바르게 호출되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        training_data_path = tmp_path / "training_data.jsonl"
        training_data_path.touch()
        expected_adapter_path = tmp_path / "adapter"

        mock_loader_cls = mocker.patch("slm_factory.trainer.DataLoader")
        mock_trainer_cls = mocker.patch("slm_factory.trainer.LoRATrainer")
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.train.return_value = expected_adapter_path

        result = pipeline.step_train(training_data_path)

        assert result == expected_adapter_path


# ---------------------------------------------------------------------------
# step_export
# ---------------------------------------------------------------------------


class TestStepExport:
    """Pipeline.step_export 메서드의 테스트입니다."""

    def test_HFExporter_및_OllamaExporter_호출(self, make_config, tmp_path, mocker):
        """HFExporter와 OllamaExporter가 올바르게 호출되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        expected_export_path = tmp_path / "export"

        mock_hf_cls = mocker.patch("slm_factory.exporter.HFExporter")
        mock_hf = mock_hf_cls.return_value
        mock_hf.export.return_value = expected_export_path

        mock_ollama_cls = mocker.patch("slm_factory.exporter.OllamaExporter")
        mock_ollama = mock_ollama_cls.return_value
        mock_ollama.export.return_value = expected_export_path

        result = pipeline.step_export(adapter_path)

        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# step_score
# ---------------------------------------------------------------------------


class TestStepScoreDisabled:
    """Pipeline.step_score 비활성화 테스트입니다."""

    def test_비활성화시_입력_그대로_반환(self, make_config, make_qa_pair, tmp_path):
        """scoring.enabled=False이면 입력 쌍을 그대로 반환하는지 확인합니다."""
        config = make_config(
            scoring={"enabled": False},
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair(), make_qa_pair(question="두 번째")]
        result = pipeline.step_score(pairs)

        assert result is pairs


# ---------------------------------------------------------------------------
# step_augment
# ---------------------------------------------------------------------------


class TestStepAugmentDisabled:
    """Pipeline.step_augment 비활성화 테스트입니다."""

    def test_비활성화시_입력_그대로_반환(self, make_config, make_qa_pair, tmp_path):
        """augment.enabled=False이면 입력 쌍을 그대로 반환하는지 확인합니다."""
        config = make_config(
            augment={"enabled": False},
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        result = pipeline.step_augment(pairs)

        assert result is pairs


# ---------------------------------------------------------------------------
# step_analyze
# ---------------------------------------------------------------------------


class TestStepAnalyzeDisabled:
    """Pipeline.step_analyze 비활성화 테스트입니다."""

    def test_비활성화시_None_반환(self, make_config, make_qa_pair, tmp_path):
        """analyzer.enabled=False이면 None을 반환하는지 확인합니다."""
        config = make_config(
            analyzer={"enabled": False},
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        result = pipeline.step_analyze(pairs)

        assert result is None


# ---------------------------------------------------------------------------
# run (전체 파이프라인)
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Pipeline.run 전체 실행 테스트입니다."""

    def test_전체_파이프라인_실행(self, make_config, tmp_path, mocker):
        """모든 단계를 mock하여 전체 파이프라인이 정상적으로 실행되는지 확인합니다."""
        pipeline = _make_pipeline(make_config, tmp_path)

        mock_docs = [MagicMock()]
        mock_pairs = [MagicMock()]
        mock_validated = [MagicMock()]
        mock_training_path = tmp_path / "training_data.jsonl"
        mock_adapter_path = tmp_path / "adapter"
        mock_export_path = tmp_path / "export"

        mocker.patch.object(pipeline, "step_parse", return_value=mock_docs)
        mocker.patch.object(pipeline, "step_generate", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_validate", return_value=mock_validated)
        mocker.patch.object(pipeline, "step_score", return_value=mock_validated)
        mocker.patch.object(pipeline, "step_augment", return_value=mock_validated)
        mocker.patch.object(pipeline, "step_analyze")
        mocker.patch.object(pipeline, "step_convert", return_value=mock_training_path)
        mocker.patch.object(pipeline, "step_train", return_value=mock_adapter_path)
        mocker.patch.object(pipeline, "step_export", return_value=mock_export_path)

        result = pipeline.run()

        pipeline.step_parse.assert_called_once()
        pipeline.step_generate.assert_called_once_with(mock_docs)
        pipeline.step_validate.assert_called_once()
        pipeline.step_convert.assert_called_once_with(mock_validated)
        pipeline.step_train.assert_called_once_with(mock_training_path)
        pipeline.step_export.assert_called_once_with(mock_adapter_path)
        assert result == mock_export_path


# ---------------------------------------------------------------------------
# step_eval
# ---------------------------------------------------------------------------


class TestStepEval:
    """Pipeline.step_eval 메서드의 테스트입니다."""

    def test_비활성화시_빈_리스트_반환(self, make_config, make_qa_pair, tmp_path):
        """eval.enabled=False일 때 빈 리스트를 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            eval={"enabled": False},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        result = pipeline.step_eval(pairs, "test-model")

        assert result == []

    def test_정상_평가_실행(self, make_config, make_qa_pair, tmp_path, mocker):
        """ModelEvaluator를 mock하여 평가 결과를 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            eval={"enabled": True},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        mock_results = [MagicMock()]

        mock_evaluator_cls = mocker.patch("slm_factory.evaluator.ModelEvaluator")
        mock_evaluator = mock_evaluator_cls.return_value
        mock_evaluator.evaluate.return_value = mock_results
        mock_evaluator.save_results = MagicMock()
        mock_evaluator.print_summary = MagicMock()

        result = pipeline.step_eval(pairs, "test-model")

        assert result == mock_results
        mock_evaluator.evaluate.assert_called_once_with(pairs, "test-model")

    def test_결과_파일_저장_호출(self, make_config, make_qa_pair, tmp_path, mocker):
        """evaluator.save_results가 올바른 경로로 호출되는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            eval={"enabled": True},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        mock_results = [MagicMock()]

        mock_evaluator_cls = mocker.patch("slm_factory.evaluator.ModelEvaluator")
        mock_evaluator = mock_evaluator_cls.return_value
        mock_evaluator.evaluate.return_value = mock_results
        mock_evaluator.save_results = MagicMock()
        mock_evaluator.print_summary = MagicMock()

        pipeline.step_eval(pairs, "test-model")

        expected_path = pipeline.output_dir / config.eval.output_file
        mock_evaluator.save_results.assert_called_once_with(mock_results, expected_path)


# ---------------------------------------------------------------------------
# step_gguf_export
# ---------------------------------------------------------------------------


class TestStepGgufExport:
    """Pipeline.step_gguf_export 메서드의 테스트입니다."""

    def test_비활성화시_model_dir_반환(self, make_config, tmp_path):
        """gguf_export.enabled=False일 때 model_dir을 그대로 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            gguf_export={"enabled": False},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        model_dir = tmp_path / "model"
        result = pipeline.step_gguf_export(model_dir)

        assert result == model_dir

    def test_정상_변환_실행(self, make_config, tmp_path, mocker):
        """GGUFExporter를 mock하여 GGUF 경로를 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            gguf_export={"enabled": True},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        model_dir = tmp_path / "model"
        expected_gguf = tmp_path / "model.gguf"

        mock_exporter_cls = mocker.patch("slm_factory.exporter.gguf_export.GGUFExporter")
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = expected_gguf

        result = pipeline.step_gguf_export(model_dir)

        assert result == expected_gguf

    def test_exporter_export_호출(self, make_config, tmp_path, mocker):
        """exporter.export가 model_dir로 호출되는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            gguf_export={"enabled": True},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        model_dir = tmp_path / "model"

        mock_exporter_cls = mocker.patch("slm_factory.exporter.gguf_export.GGUFExporter")
        mock_exporter = mock_exporter_cls.return_value
        mock_exporter.export.return_value = tmp_path / "model.gguf"

        pipeline.step_gguf_export(model_dir)

        mock_exporter.export.assert_called_once_with(model_dir)


# ---------------------------------------------------------------------------
# step_dialogue
# ---------------------------------------------------------------------------


class TestStepDialogue:
    """Pipeline.step_dialogue 메서드의 테스트입니다."""

    def test_비활성화시_빈_리스트_반환(self, make_config, make_qa_pair, tmp_path):
        """dialogue.enabled=False일 때 빈 리스트를 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            dialogue={"enabled": False},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        result = pipeline.step_dialogue(pairs)

        assert result == []

    def test_정상_대화_생성(self, make_config, make_qa_pair, tmp_path, mocker):
        """DialogueGenerator를 mock하여 대화 목록을 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            dialogue={"enabled": True},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        mock_dialogues = [MagicMock()]

        mocker.patch("slm_factory.teacher.create_teacher", return_value=MagicMock())
        mock_generator_cls = mocker.patch(
            "slm_factory.teacher.dialogue_generator.DialogueGenerator"
        )
        mock_generator = mock_generator_cls.return_value
        mocker.patch("slm_factory.pipeline.asyncio.run", return_value=mock_dialogues)
        mock_generator.save_dialogues = MagicMock()

        result = pipeline.step_dialogue(pairs)

        assert result == mock_dialogues

    def test_대화_파일_저장(self, make_config, make_qa_pair, tmp_path, mocker):
        """generator.save_dialogues가 올바른 경로로 호출되는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            dialogue={"enabled": True},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        mock_dialogues = [MagicMock()]

        mocker.patch("slm_factory.teacher.create_teacher", return_value=MagicMock())
        mock_generator_cls = mocker.patch(
            "slm_factory.teacher.dialogue_generator.DialogueGenerator"
        )
        mock_generator = mock_generator_cls.return_value
        mocker.patch("slm_factory.pipeline.asyncio.run", return_value=mock_dialogues)
        mock_generator.save_dialogues = MagicMock()

        pipeline.step_dialogue(pairs)

        expected_path = pipeline.output_dir / "dialogues.json"
        mock_generator.save_dialogues.assert_called_once_with(mock_dialogues, expected_path)


# ---------------------------------------------------------------------------
# step_compare
# ---------------------------------------------------------------------------


class TestStepCompare:
    """Pipeline.step_compare 메서드의 테스트입니다."""

    def test_비활성화시_빈_리스트_반환(self, make_config, make_qa_pair, tmp_path):
        """compare.enabled=False일 때 빈 리스트를 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            compare={"enabled": False},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        result = pipeline.step_compare(pairs)

        assert result == []

    def test_정상_비교_실행(self, make_config, make_qa_pair, tmp_path, mocker):
        """ModelComparator를 mock하여 비교 결과를 반환하는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            compare={"enabled": True, "base_model": "base", "finetuned_model": "ft"},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        mock_results = [MagicMock()]

        mock_comparator_cls = mocker.patch("slm_factory.comparator.ModelComparator")
        mock_comparator = mock_comparator_cls.return_value
        mock_comparator.compare.return_value = mock_results
        mock_comparator.save_results = MagicMock()
        mock_comparator.print_summary = MagicMock()

        result = pipeline.step_compare(pairs)

        assert result == mock_results
        mock_comparator.compare.assert_called_once_with(pairs)

    def test_결과_파일_저장_호출(self, make_config, make_qa_pair, tmp_path, mocker):
        """comparator.save_results가 올바른 경로로 호출되는지 확인합니다."""
        config = make_config(
            paths={"output": str(tmp_path / "output"), "documents": str(tmp_path / "docs")},
            compare={"enabled": True, "base_model": "base", "finetuned_model": "ft"},
        )
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(config)

        pairs = [make_qa_pair()]
        mock_results = [MagicMock()]

        mock_comparator_cls = mocker.patch("slm_factory.comparator.ModelComparator")
        mock_comparator = mock_comparator_cls.return_value
        mock_comparator.compare.return_value = mock_results
        mock_comparator.save_results = MagicMock()
        mock_comparator.print_summary = MagicMock()

        pipeline.step_compare(pairs)

        expected_path = pipeline.output_dir / config.compare.output_file
        mock_comparator.save_results.assert_called_once_with(mock_results, expected_path)
