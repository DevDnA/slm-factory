"""ModelEvaluator 테스트 — mock 기반 (httpx + evaluate 패키지 모킹)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slm_factory.config import EvalConfig, SLMConfig, TeacherConfig
from slm_factory.evaluator import ModelEvaluator
from slm_factory.models import EvalResult, QAPair


@pytest.fixture
def slm_config():
    return SLMConfig(
        teacher=TeacherConfig(api_base="http://localhost:11434", timeout=30, max_concurrency=2),
        eval=EvalConfig(enabled=True, metrics=["bleu", "rouge"], max_samples=10),
    )


@pytest.fixture
def evaluator(slm_config):
    return ModelEvaluator(slm_config)


@pytest.fixture
def sample_pairs():
    return [
        QAPair(question="한국의 수도는?", answer="서울입니다.", source_doc="geo.pdf", category="지리"),
        QAPair(question="1+1은?", answer="2입니다.", source_doc="math.pdf", category="수학"),
    ]


@pytest.fixture
def mock_bleu():
    m = MagicMock()
    m.compute.return_value = {"bleu": 0.75}
    return m


@pytest.fixture
def mock_rouge():
    m = MagicMock()
    m.compute.return_value = {"rouge1": 0.8, "rouge2": 0.6, "rougeL": 0.7}
    return m


def _mock_httpx_response(text: str):
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    return resp


class TestComputeScores:
    def test_bleu_and_rouge(self, evaluator, mock_bleu, mock_rouge):
        with (
            patch("slm_factory.evaluator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.evaluator._load_rouge", return_value=mock_rouge),
        ):
            scores = evaluator._compute_scores("서울입니다.", "서울입니다.")

        assert "bleu" in scores
        assert scores["bleu"] == 0.75
        assert "rouge1" in scores
        assert "rougeL" in scores

    def test_bleu_only(self, slm_config, mock_bleu):
        slm_config.eval.metrics = ["bleu"]
        ev = ModelEvaluator(slm_config)
        with patch("slm_factory.evaluator._load_bleu", return_value=mock_bleu):
            scores = ev._compute_scores("ref", "gen")

        assert "bleu" in scores
        assert "rouge1" not in scores

    def test_rouge_only(self, slm_config, mock_rouge):
        slm_config.eval.metrics = ["rouge"]
        ev = ModelEvaluator(slm_config)
        with patch("slm_factory.evaluator._load_rouge", return_value=mock_rouge):
            scores = ev._compute_scores("ref", "gen")

        assert "rouge1" in scores
        assert "bleu" not in scores


class TestEvaluate:
    def test_returns_eval_results(self, evaluator, sample_pairs, mock_bleu, mock_rouge):
        mock_response = _mock_httpx_response("서울입니다.")

        with (
            patch("slm_factory.evaluator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.evaluator._load_rouge", return_value=mock_rouge),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = evaluator.evaluate(sample_pairs, "test-model")

        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)
        assert results[0].question == "한국의 수도는?"
        assert results[0].generated_answer == "서울입니다."
        assert "bleu" in results[0].scores

    def test_empty_input(self, evaluator):
        results = evaluator.evaluate([], "test-model")
        assert results == []

    def test_max_samples_limit(self, slm_config, mock_bleu, mock_rouge):
        slm_config.eval.max_samples = 2
        ev = ModelEvaluator(slm_config)
        pairs = [
            QAPair(question=f"q{i}?", answer=f"a{i}", source_doc="d.pdf", category="c")
            for i in range(5)
        ]

        mock_response = _mock_httpx_response("answer")

        with (
            patch("slm_factory.evaluator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.evaluator._load_rouge", return_value=mock_rouge),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = ev.evaluate(pairs, "test-model")

        assert len(results) == 2


class TestSaveResults:
    def test_writes_valid_json(self, evaluator, tmp_path):
        results = [
            EvalResult(
                question="q1?",
                reference_answer="a1",
                generated_answer="g1",
                scores={"bleu": 0.5, "rouge1": 0.6},
            ),
            EvalResult(
                question="q2?",
                reference_answer="a2",
                generated_answer="g2",
                scores={"bleu": 0.7, "rouge1": 0.8},
            ),
        ]

        out_path = tmp_path / "results" / "eval.json"
        evaluator.save_results(results, out_path)

        assert out_path.is_file()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["question"] == "q1?"
        assert data[0]["scores"]["bleu"] == 0.5

    def test_empty_results(self, evaluator, tmp_path):
        out_path = tmp_path / "empty.json"
        evaluator.save_results([], out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data == []


class TestPrintSummary:
    def test_runs_without_error(self, evaluator, capsys):
        results = [
            EvalResult(question="q?", reference_answer="a", generated_answer="g", scores={"bleu": 0.5}),
        ]
        evaluator.print_summary(results)
        out = capsys.readouterr().out
        assert "평가 결과 요약" in out
        assert "1건 평가 완료" in out

    def test_empty_results(self, evaluator, capsys):
        evaluator.print_summary([])
        out = capsys.readouterr().out
        assert "평가 결과가 없습니다" in out

    def test_multiple_metrics(self, evaluator, capsys):
        results = [
            EvalResult(question="q1?", reference_answer="a1", generated_answer="g1", scores={"bleu": 0.5, "rouge1": 0.6}),
            EvalResult(question="q2?", reference_answer="a2", generated_answer="g2", scores={"bleu": 0.7, "rouge1": 0.8}),
        ]
        evaluator.print_summary(results)
        out = capsys.readouterr().out
        assert "bleu" in out
        assert "rouge1" in out
        assert "2건 평가 완료" in out
