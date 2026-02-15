"""TUI 대시보드 데이터 로직 테스트입니다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_factory.tui.dashboard import (
    PipelineSnapshot,
    StageInfo,
    _extract_compare_summary,
    _extract_eval_summary,
    scan_pipeline,
)


# ---------------------------------------------------------------------------
# TestStageInfo
# ---------------------------------------------------------------------------


class TestStageInfo:

    def test_defaults(self):
        info = StageInfo(name="parse", display_name="문서 파싱", filename="parsed_documents.json")
        assert info.exists is False
        assert info.count == 0
        assert info.unit == "건"

    def test_custom_values(self):
        info = StageInfo(
            name="generate",
            display_name="QA 생성",
            filename="qa_alpaca.json",
            exists=True,
            count=42,
            unit="쌍",
        )
        assert info.exists is True
        assert info.count == 42
        assert info.unit == "쌍"


# ---------------------------------------------------------------------------
# TestPipelineSnapshot
# ---------------------------------------------------------------------------


class TestPipelineSnapshot:

    def test_empty_snapshot(self):
        snap = PipelineSnapshot()
        assert snap.stages == []
        assert snap.eval_summary == {}
        assert snap.compare_summary == {}

    def test_snapshot_with_stages(self):
        stages = [
            StageInfo(name="parse", display_name="문서 파싱", filename="parsed_documents.json", exists=True, count=5),
        ]
        snap = PipelineSnapshot(stages=stages)
        assert len(snap.stages) == 1
        assert snap.stages[0].count == 5


# ---------------------------------------------------------------------------
# TestScanPipeline
# ---------------------------------------------------------------------------


class TestScanPipeline:

    def test_empty_directory(self, tmp_path: Path):
        snap = scan_pipeline(tmp_path)
        assert len(snap.stages) == 11
        assert all(not s.exists for s in snap.stages)

    def test_parsed_documents(self, tmp_path: Path):
        docs = [{"doc_id": "a.pdf", "content": "hello"}]
        (tmp_path / "parsed_documents.json").write_text(json.dumps(docs), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        parse_stage = next(s for s in snap.stages if s.name == "parse")
        assert parse_stage.exists is True
        assert parse_stage.count == 1

    def test_qa_alpaca_json(self, tmp_path: Path):
        pairs = [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}, {"q": "Q3", "a": "A3"}]
        (tmp_path / "qa_alpaca.json").write_text(json.dumps(pairs), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        gen_stage = next(s for s in snap.stages if s.name == "generate")
        assert gen_stage.exists is True
        assert gen_stage.count == 3

    def test_qa_scored_json(self, tmp_path: Path):
        scored = [{"q": "Q", "score": 0.9}] * 5
        (tmp_path / "qa_scored.json").write_text(json.dumps(scored), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        score_stage = next(s for s in snap.stages if s.name == "score")
        assert score_stage.exists is True
        assert score_stage.count == 5

    def test_training_data_jsonl(self, tmp_path: Path):
        lines = ['{"text": "line1"}\n', '{"text": "line2"}\n', '{"text": "line3"}\n']
        (tmp_path / "training_data.jsonl").write_text("".join(lines), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        convert_stage = next(s for s in snap.stages if s.name == "convert")
        assert convert_stage.exists is True
        assert convert_stage.count == 3

    def test_checkpoint_adapter_directory(self, tmp_path: Path):
        (tmp_path / "checkpoints" / "adapter").mkdir(parents=True)

        snap = scan_pipeline(tmp_path)
        train_stage = next(s for s in snap.stages if s.name == "train")
        assert train_stage.exists is True

    def test_merged_model_directory(self, tmp_path: Path):
        (tmp_path / "merged_model").mkdir()

        snap = scan_pipeline(tmp_path)
        export_stage = next(s for s in snap.stages if s.name == "export")
        assert export_stage.exists is True

    def test_eval_results_populates_summary(self, tmp_path: Path):
        results = [
            {"question": "Q1", "scores": {"bleu": 0.8, "rouge": 0.6}},
            {"question": "Q2", "scores": {"bleu": 0.6, "rouge": 0.4}},
        ]
        (tmp_path / "eval_results.json").write_text(json.dumps(results), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        assert snap.eval_summary["bleu"] == 0.7
        assert snap.eval_summary["rouge"] == 0.5

    def test_compare_results_populates_summary(self, tmp_path: Path):
        data = {"summary": {"win": 3, "lose": 1, "tie": 1}, "results": []}
        (tmp_path / "compare_results.json").write_text(json.dumps(data), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        assert snap.compare_summary["win"] == 3
        assert snap.compare_summary["lose"] == 1

    def test_data_analysis_json(self, tmp_path: Path):
        analysis = [{"metric": "avg_len", "value": 120}] * 7
        (tmp_path / "data_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        analyze_stage = next(s for s in snap.stages if s.name == "analyze")
        assert analyze_stage.exists is True
        assert analyze_stage.count == 7

    def test_dialogues_json(self, tmp_path: Path):
        dialogues = [{"turns": []}] * 4
        (tmp_path / "dialogues.json").write_text(json.dumps(dialogues), encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        dialogue_stage = next(s for s in snap.stages if s.name == "dialogue")
        assert dialogue_stage.exists is True
        assert dialogue_stage.count == 4

    def test_full_pipeline(self, tmp_path: Path):
        (tmp_path / "parsed_documents.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_alpaca.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_scored.json").write_text("[]", encoding="utf-8")
        (tmp_path / "qa_augmented.json").write_text("[]", encoding="utf-8")
        (tmp_path / "data_analysis.json").write_text("[]", encoding="utf-8")
        (tmp_path / "training_data.jsonl").write_text("", encoding="utf-8")
        (tmp_path / "checkpoints" / "adapter").mkdir(parents=True)
        (tmp_path / "merged_model").mkdir()
        (tmp_path / "eval_results.json").write_text("[]", encoding="utf-8")
        (tmp_path / "compare_results.json").write_text("{}", encoding="utf-8")
        (tmp_path / "dialogues.json").write_text("[]", encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        assert all(s.exists for s in snap.stages)

    def test_malformed_json_file(self, tmp_path: Path):
        (tmp_path / "parsed_documents.json").write_text("not json!", encoding="utf-8")

        snap = scan_pipeline(tmp_path)
        parse_stage = next(s for s in snap.stages if s.name == "parse")
        assert parse_stage.exists is True
        assert parse_stage.count == 0

    def test_nonexistent_directory(self, tmp_path: Path):
        snap = scan_pipeline(tmp_path / "does_not_exist")
        assert len(snap.stages) == 11
        assert all(not s.exists for s in snap.stages)


# ---------------------------------------------------------------------------
# TestMetricsExtraction
# ---------------------------------------------------------------------------


class TestMetricsExtraction:

    def test_eval_summary_average(self, tmp_path: Path):
        results = [
            {"scores": {"bleu": 0.5}},
            {"scores": {"bleu": 0.9}},
        ]
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(results), encoding="utf-8")
        summary = _extract_eval_summary(p)
        assert summary["bleu"] == 0.7

    def test_eval_summary_missing_file(self, tmp_path: Path):
        assert _extract_eval_summary(tmp_path / "missing.json") == {}

    def test_eval_summary_empty_list(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text("[]", encoding="utf-8")
        assert _extract_eval_summary(p) == {}

    def test_eval_summary_non_list(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        assert _extract_eval_summary(p) == {}

    def test_eval_summary_skips_non_numeric(self, tmp_path: Path):
        results = [{"scores": {"bleu": 0.8, "note": "good"}}]
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(results), encoding="utf-8")
        summary = _extract_eval_summary(p)
        assert "bleu" in summary
        assert "note" not in summary

    def test_compare_summary_dict_with_summary(self, tmp_path: Path):
        data = {"summary": {"win": 5, "lose": 2}}
        p = tmp_path / "compare.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        summary = _extract_compare_summary(p)
        assert summary == {"win": 5, "lose": 2}

    def test_compare_summary_list(self, tmp_path: Path):
        p = tmp_path / "compare.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        summary = _extract_compare_summary(p)
        assert summary == {"total": 3}

    def test_compare_summary_missing_file(self, tmp_path: Path):
        assert _extract_compare_summary(tmp_path / "missing.json") == {}

    def test_compare_summary_dict_without_summary_key(self, tmp_path: Path):
        data = {"result_a": 1, "result_b": 2}
        p = tmp_path / "compare.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        summary = _extract_compare_summary(p)
        assert summary == {"total": 2}

    def test_eval_summary_malformed_json(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text("broken!", encoding="utf-8")
        assert _extract_eval_summary(p) == {}

    def test_compare_summary_malformed_json(self, tmp_path: Path):
        p = tmp_path / "compare.json"
        p.write_text("broken!", encoding="utf-8")
        assert _extract_compare_summary(p) == {}
