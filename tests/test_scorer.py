"""QualityScorer 테스트 — mock 기반 (teacher.agenerate 모킹)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slm_factory.config import ScoringConfig, TeacherConfig
from slm_factory.models import QAPair
from slm_factory.scorer import QualityScorer


@pytest.fixture
def teacher_config():
    return TeacherConfig(backend="openai")


@pytest.fixture
def scoring_config():
    return ScoringConfig(enabled=True, threshold=3.0)


@pytest.fixture
def mock_teacher():
    teacher = MagicMock()
    teacher.agenerate = AsyncMock()
    return teacher


@pytest.fixture
def scorer(mock_teacher, scoring_config, teacher_config):
    return QualityScorer(mock_teacher, scoring_config, teacher_config)


@pytest.fixture
def sample_pair():
    return QAPair(
        question="테스트 질문입니다?",
        answer="이것은 테스트 답변입니다.",
        source_doc="test.pdf",
        category="general",
    )


class TestBuildScoringPrompt:
    def test_contains_question_and_answer(self, scorer, sample_pair):
        prompt = scorer._build_scoring_prompt(sample_pair)
        assert sample_pair.question in prompt
        assert sample_pair.answer in prompt

    def test_contains_scoring_criteria(self, scorer, sample_pair):
        prompt = scorer._build_scoring_prompt(sample_pair)
        assert "1~5점" in prompt
        assert "JSON" in prompt


    def test_source_text_포함_시_원본_문서_섹션_존재(self, scorer, sample_pair):
        prompt = scorer._build_scoring_prompt(sample_pair, source_text="문서 내용입니다.")
        assert "원본 문서" in prompt
        assert "문서 내용입니다." in prompt

    def test_source_text_없으면_원본_문서_섹션_없음(self, scorer, sample_pair):
        prompt = scorer._build_scoring_prompt(sample_pair, source_text="")
        assert "원본 문서" not in prompt

    def test_source_text_3000자_초과_시_잘림(self, scorer, sample_pair):
        long_text = "가" * 4000
        prompt = scorer._build_scoring_prompt(sample_pair, source_text=long_text)
        assert "가" * 3000 in prompt
        assert "가" * 3001 not in prompt

    def test_환각_탐지_기준_포함(self, scorer, sample_pair):
        prompt = scorer._build_scoring_prompt(sample_pair, source_text="참고 문서 내용")
        assert "환각" in prompt or "근거" in prompt


class TestParseScore:
    def test_valid_json(self, scorer):
        text = '{"score": 4, "reason": "좋은 답변입니다"}'
        result = scorer._parse_score(text)
        assert result == (4, "좋은 답변입니다")

    def test_json_with_code_fence(self, scorer):
        text = '```json\n{"score": 5, "reason": "완벽합니다"}\n```'
        result = scorer._parse_score(text)
        assert result == (5, "완벽합니다")

    def test_number_only_fallback(self, scorer):
        text = "점수는 4점입니다."
        result = scorer._parse_score(text)
        assert isinstance(result, tuple)
        assert result[0] == 4

    def test_parse_failure_returns_none(self, scorer):
        text = "이것은 점수가 아닙니다 zero"
        result = scorer._parse_score(text)
        assert result is None

    def test_out_of_range_score_fallback(self, scorer):
        text = '{"score": 9, "reason": "범위 초과"}'
        result = scorer._parse_score(text)
        assert result is None or result[0] <= 5


class TestScoreAll:
    async def test_threshold_filtering(self, mock_teacher, scoring_config, teacher_config):
        scoring_config.threshold = 4.0
        scorer = QualityScorer(mock_teacher, scoring_config, teacher_config)

        pairs = [
            QAPair(question="좋은 질문?", answer="좋은 답변", source_doc="a.pdf", category="g"),
            QAPair(question="나쁜 질문?", answer="나쁜 답변", source_doc="b.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(side_effect=[
            json.dumps({"score": 5, "reason": "훌륭합니다"}),
            json.dumps({"score": 2, "reason": "부족합니다"}),
        ])

        accepted, filtered = await scorer.score_all(pairs)
        assert len(accepted) == 1
        assert len(filtered) == 1
        assert accepted[0].question == "좋은 질문?"

    async def test_all_pass(self, mock_teacher, scoring_config, teacher_config):
        scoring_config.threshold = 1.0
        scorer = QualityScorer(mock_teacher, scoring_config, teacher_config)

        pairs = [
            QAPair(question="q1?", answer="a1", source_doc="a.pdf", category="g"),
            QAPair(question="q2?", answer="a2", source_doc="a.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(side_effect=[
            json.dumps({"score": 3, "reason": "ok"}),
            json.dumps({"score": 4, "reason": "good"}),
        ])

        accepted, filtered = await scorer.score_all(pairs)
        assert len(accepted) == 2
        assert len(filtered) == 0

    async def test_parse_failure_defaults_to_0(self, mock_teacher, scoring_config, teacher_config):
        """파싱 실패 시 기본 점수 0으로 필터링됩니다."""
        scoring_config.threshold = 3.0
        scorer = QualityScorer(mock_teacher, scoring_config, teacher_config)

        pairs = [QAPair(question="q?", answer="a", source_doc="a.pdf", category="g")]
        mock_teacher.agenerate = AsyncMock(return_value="unparseable garbage no digits")

        accepted, filtered = await scorer.score_all(pairs)
        assert len(accepted) == 0
        assert len(filtered) == 1

    async def test_source_texts_전달(self, mock_teacher, scoring_config, teacher_config):
        scoring_config.threshold = 1.0
        scorer = QualityScorer(mock_teacher, scoring_config, teacher_config)

        pairs = [
            QAPair(question="q1?", answer="a1", source_doc="doc1.pdf", category="g"),
            QAPair(question="q2?", answer="a2", source_doc="doc2.pdf", category="g"),
        ]
        source_texts = {"doc1.pdf": "문서1 내용", "doc2.pdf": "문서2 내용"}

        mock_teacher.agenerate = AsyncMock(side_effect=[
            json.dumps({"score": 5, "reason": "ok"}),
            json.dumps({"score": 5, "reason": "ok"}),
        ])

        with patch.object(scorer, "_build_scoring_prompt", wraps=scorer._build_scoring_prompt) as mock_build:
            await scorer.score_all(pairs, source_texts=source_texts)
            assert mock_build.call_count == 2
            calls = mock_build.call_args_list
            assert calls[0].kwargs.get("source_text") == "문서1 내용" or (
                len(calls[0].args) > 1 and calls[0].args[1] == "문서1 내용"
            )
            assert calls[1].kwargs.get("source_text") == "문서2 내용" or (
                len(calls[1].args) > 1 and calls[1].args[1] == "문서2 내용"
            )
