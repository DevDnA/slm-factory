"""ModelEvaluator 테스트 — mock 기반 (httpx + evaluate 패키지 모킹)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slm_factory.config import EvalConfig, ProjectConfig, SLMConfig, TeacherConfig
from slm_factory.evaluator import ModelEvaluator, _preprocess_for_metrics
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


class TestKoreanTokenizer:
    """한국어 형태소 분석 토크나이저 관련 테스트입니다."""

    def test_전처리_토크나이저_없음(self):
        """토크나이저가 None이면 원본 텍스트를 반환하는지 확인합니다."""
        text = "한국어 텍스트입니다."
        result = _preprocess_for_metrics(text, korean_tokenizer=None)
        assert result == text

    def test_전처리_토크나이저_적용(self):
        """토크나이저가 제공되면 형태소 분리된 텍스트를 반환하는지 확인합니다."""
        def mock_tokenizer(text):
            return ["한국어", "텍스트", "입니다", "."]

        result = _preprocess_for_metrics("한국어 텍스트입니다.", korean_tokenizer=mock_tokenizer)
        assert result == "한국어 텍스트 입니다 ."

    def test_한국어_프로젝트_토크나이저_사용(self, mock_bleu, mock_rouge):
        """language='ko'일 때 한국어 토크나이저가 적용되는지 확인합니다."""
        config = SLMConfig(
            project=ProjectConfig(language="ko"),
            teacher=TeacherConfig(api_base="http://localhost:11434", timeout=30, max_concurrency=2),
            eval=EvalConfig(enabled=True, metrics=["bleu", "rouge"], max_samples=10),
        )
        ev = ModelEvaluator(config)

        mock_tokenizer = MagicMock(side_effect=lambda text: text.split())
        with (
            patch("slm_factory.evaluator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.evaluator._load_rouge", return_value=mock_rouge),
            patch("slm_factory.evaluator._load_korean_tokenizer", return_value=mock_tokenizer),
        ):
            ev._compute_scores("서울입니다", "서울입니다")

        assert mock_tokenizer.call_count >= 2

    def test_영어_프로젝트_토크나이저_미사용(self, mock_bleu, mock_rouge):
        """language='en'일 때 한국어 토크나이저가 사용되지 않는지 확인합니다."""
        config = SLMConfig(
            project=ProjectConfig(language="en"),
            teacher=TeacherConfig(api_base="http://localhost:11434", timeout=30, max_concurrency=2),
            eval=EvalConfig(enabled=True, metrics=["bleu", "rouge"], max_samples=10),
        )
        ev = ModelEvaluator(config)

        with (
            patch("slm_factory.evaluator._load_bleu", return_value=mock_bleu),
            patch("slm_factory.evaluator._load_rouge", return_value=mock_rouge),
            patch("slm_factory.evaluator._load_korean_tokenizer") as mock_load,
        ):
            ev._compute_scores("hello", "hello")

        mock_load.assert_not_called()

    def test_빈_텍스트_전처리(self):
        """빈 텍스트에 대해 전처리가 정상 동작하는지 확인합니다."""
        def mock_tokenizer(text):
            return []

        result = _preprocess_for_metrics("", korean_tokenizer=mock_tokenizer)
        assert result == ""


class TestCheckQualityGate:
    """check_quality_gate 메서드 테스트 — 품질 임계값 기반 통과/실패 판정."""

    def test_임계값_통과(self, evaluator):
        """모든 메트릭이 임계값을 넘으면 (True, 평균값)을 반환합니다."""
        results = [
            EvalResult(question="q1?", reference_answer="a1", generated_answer="g1", scores={"bleu": 0.5, "rougeL": 0.6}),
            EvalResult(question="q2?", reference_answer="a2", generated_answer="g2", scores={"bleu": 0.3, "rougeL": 0.4}),
        ]
        thresholds = {"bleu": 0.3, "rougeL": 0.4}
        passed, averages = evaluator.check_quality_gate(results, thresholds)

        assert passed is True
        assert averages["bleu"] == pytest.approx(0.4)
        assert averages["rougeL"] == pytest.approx(0.5)

    def test_임계값_미달(self, evaluator):
        """메트릭 평균이 임계값 미만이면 (False, 평균값)을 반환합니다."""
        results = [
            EvalResult(question="q1?", reference_answer="a1", generated_answer="g1", scores={"bleu": 0.1, "rougeL": 0.2}),
            EvalResult(question="q2?", reference_answer="a2", generated_answer="g2", scores={"bleu": 0.2, "rougeL": 0.1}),
        ]
        thresholds = {"bleu": 0.5, "rougeL": 0.5}
        passed, averages = evaluator.check_quality_gate(results, thresholds)

        assert passed is False
        assert "bleu" in averages
        assert "rougeL" in averages

    def test_빈_결과(self, evaluator):
        """빈 결과 리스트를 전달하면 (False, {})를 반환합니다."""
        passed, averages = evaluator.check_quality_gate([], {"bleu": 0.3})

        assert passed is False
        assert averages == {}

    def test_일부_메트릭만_미달(self, evaluator):
        """일부 메트릭만 임계값 미달이어도 전체 게이트가 실패합니다."""
        results = [
            EvalResult(question="q1?", reference_answer="a1", generated_answer="g1", scores={"bleu": 0.8, "rougeL": 0.1}),
            EvalResult(question="q2?", reference_answer="a2", generated_answer="g2", scores={"bleu": 0.9, "rougeL": 0.2}),
        ]
        thresholds = {"bleu": 0.5, "rougeL": 0.5}
        passed, averages = evaluator.check_quality_gate(results, thresholds)

        assert passed is False
        assert averages["bleu"] == pytest.approx(0.85)
        assert averages["rougeL"] == pytest.approx(0.15)
