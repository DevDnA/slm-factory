"""DialogueGenerator 테스트 — mock 기반 (teacher.agenerate 모킹)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from slm_factory.config import DialogueConfig, TeacherConfig
from slm_factory.models import DialogueTurn, MultiTurnDialogue, QAPair
from slm_factory.teacher.dialogue_generator import DialogueGenerator


@pytest.fixture
def teacher_config():
    return TeacherConfig(backend="openai")


@pytest.fixture
def dialogue_config():
    return DialogueConfig(enabled=True, min_turns=2, max_turns=5)


@pytest.fixture
def mock_teacher():
    teacher = MagicMock()
    teacher.agenerate = AsyncMock()
    return teacher


@pytest.fixture
def generator(mock_teacher, dialogue_config, teacher_config):
    return DialogueGenerator(mock_teacher, dialogue_config, teacher_config)


@pytest.fixture
def sample_pair():
    return QAPair(
        question="프로젝트의 목표는 무엇인가요?",
        answer="이 프로젝트는 도메인 특화 SLM을 만드는 것이 목표입니다.",
        source_doc="test.pdf",
        category="general",
    )


def _make_valid_response(turns=None):
    if turns is None:
        turns = [
            {"role": "user", "content": "프로젝트의 목표는 무엇인가요?"},
            {"role": "assistant", "content": "이 프로젝트는 도메인 특화 SLM을 만드는 것이 목표입니다."},
            {"role": "user", "content": "SLM이란 무엇인가요?"},
            {"role": "assistant", "content": "Small Language Model의 약자로, 소규모 언어 모델입니다."},
        ]
    return json.dumps({"turns": turns})


class TestBuildDialoguePrompt:
    def test_contains_question_and_answer(self, generator, sample_pair):
        prompt = generator._build_dialogue_prompt(sample_pair)
        assert sample_pair.question in prompt
        assert sample_pair.answer in prompt

    def test_contains_turn_range(self, generator, sample_pair):
        prompt = generator._build_dialogue_prompt(sample_pair)
        assert "2" in prompt
        assert "5" in prompt
        assert "JSON" in prompt


class TestParseDialogue:
    def test_valid_json(self, generator, sample_pair):
        text = _make_valid_response()
        result = generator._parse_dialogue(text, sample_pair)
        assert isinstance(result, MultiTurnDialogue)
        assert len(result.turns) == 4
        assert result.turns[0].role == "user"
        assert result.turns[1].role == "assistant"
        assert result.source_doc == "test.pdf"
        assert result.category == "general"

    def test_json_with_code_fence(self, generator, sample_pair):
        text = "```json\n" + _make_valid_response() + "\n```"
        result = generator._parse_dialogue(text, sample_pair)
        assert isinstance(result, MultiTurnDialogue)
        assert len(result.turns) == 4

    def test_invalid_json_returns_none(self, generator, sample_pair):
        result = generator._parse_dialogue("not json at all", sample_pair)
        assert result is None

    def test_empty_turns_returns_none(self, generator, sample_pair):
        text = json.dumps({"turns": []})
        result = generator._parse_dialogue(text, sample_pair)
        assert result is None

    def test_single_turn_returns_none(self, generator, sample_pair):
        text = json.dumps({"turns": [{"role": "user", "content": "hello"}]})
        result = generator._parse_dialogue(text, sample_pair)
        assert result is None

    def test_invalid_role_filtered(self, generator, sample_pair):
        turns = [
            {"role": "user", "content": "질문"},
            {"role": "invalid_role", "content": "무시됨"},
            {"role": "assistant", "content": "답변"},
        ]
        text = json.dumps({"turns": turns})
        result = generator._parse_dialogue(text, sample_pair)
        assert isinstance(result, MultiTurnDialogue)
        assert len(result.turns) == 2

    def test_max_turns_truncation(self, generator, sample_pair):
        generator.config = DialogueConfig(enabled=True, min_turns=2, max_turns=3)
        turns = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        text = json.dumps({"turns": turns})
        result = generator._parse_dialogue(text, sample_pair)
        assert isinstance(result, MultiTurnDialogue)
        assert len(result.turns) == 3

    def test_min_turns_enforcement(self, generator, sample_pair):
        generator.config = DialogueConfig(enabled=True, min_turns=4, max_turns=6)
        turns = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        text = json.dumps({"turns": turns})
        result = generator._parse_dialogue(text, sample_pair)
        assert result is None

    def test_empty_content_filtered(self, generator, sample_pair):
        turns = [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "후속 질문"},
            {"role": "assistant", "content": "답변"},
        ]
        text = json.dumps({"turns": turns})
        result = generator._parse_dialogue(text, sample_pair)
        assert isinstance(result, MultiTurnDialogue)
        assert all(t.content for t in result.turns)


class TestGenerateDialogue:
    async def test_returns_valid_dialogue(self, mock_teacher, generator, sample_pair):
        mock_teacher.agenerate = AsyncMock(return_value=_make_valid_response())
        result = await generator.generate_dialogue(sample_pair)
        assert isinstance(result, MultiTurnDialogue)
        assert len(result.turns) >= 2

    async def test_parse_failure_returns_none(self, mock_teacher, generator, sample_pair):
        mock_teacher.agenerate = AsyncMock(return_value="unparseable garbage")
        result = await generator.generate_dialogue(sample_pair)
        assert result is None

    async def test_ollama_backend_sends_format_json(self, sample_pair):
        teacher_config = TeacherConfig(backend="ollama")
        dialogue_config = DialogueConfig(enabled=True)
        teacher = MagicMock()
        teacher.agenerate = AsyncMock(return_value=_make_valid_response())
        gen = DialogueGenerator(teacher, dialogue_config, teacher_config)

        await gen.generate_dialogue(sample_pair)
        _, kwargs = teacher.agenerate.call_args
        assert kwargs.get("format") == "json"


class TestGenerateAll:
    async def test_multiple_pairs(self, mock_teacher, generator):
        pairs = [
            QAPair(question="q1?", answer="a1", source_doc="a.pdf", category="g"),
            QAPair(question="q2?", answer="a2", source_doc="b.pdf", category="g"),
        ]

        responses = [
            _make_valid_response([
                {"role": "user", "content": "q1?"},
                {"role": "assistant", "content": "a1"},
            ]),
            _make_valid_response([
                {"role": "user", "content": "q2?"},
                {"role": "assistant", "content": "a2"},
            ]),
        ]
        mock_teacher.agenerate = AsyncMock(side_effect=responses)

        dialogues = await generator.generate_all(pairs)
        assert len(dialogues) == 2

    async def test_partial_failures(self, mock_teacher, generator):
        pairs = [
            QAPair(question="q1?", answer="a1", source_doc="a.pdf", category="g"),
            QAPair(question="q2?", answer="a2", source_doc="b.pdf", category="g"),
        ]

        mock_teacher.agenerate = AsyncMock(side_effect=[
            _make_valid_response([
                {"role": "user", "content": "q1?"},
                {"role": "assistant", "content": "a1"},
            ]),
            "invalid response",
        ])

        dialogues = await generator.generate_all(pairs)
        assert len(dialogues) == 1

    async def test_exception_handling(self, mock_teacher, generator):
        pairs = [
            QAPair(question="q1?", answer="a1", source_doc="a.pdf", category="g"),
        ]
        mock_teacher.agenerate = AsyncMock(side_effect=RuntimeError("network error"))

        dialogues = await generator.generate_all(pairs)
        assert len(dialogues) == 0


class TestSaveDialogues:
    def test_writes_valid_json(self, generator, tmp_path):
        dialogues = [
            MultiTurnDialogue(
                turns=[
                    DialogueTurn(role="user", content="q1"),
                    DialogueTurn(role="assistant", content="a1"),
                ],
                source_doc="test.pdf",
                category="general",
            ),
        ]

        output_path = tmp_path / "dialogues.json"
        generator.save_dialogues(dialogues, output_path)

        assert output_path.is_file()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert len(data[0]["turns"]) == 2
        assert data[0]["turns"][0]["role"] == "user"
        assert data[0]["source_doc"] == "test.pdf"

    def test_creates_parent_dirs(self, generator, tmp_path):
        dialogues = [
            MultiTurnDialogue(
                turns=[
                    DialogueTurn(role="user", content="q1"),
                    DialogueTurn(role="assistant", content="a1"),
                ],
            ),
        ]

        output_path = tmp_path / "sub" / "dir" / "dialogues.json"
        generator.save_dialogues(dialogues, output_path)
        assert output_path.is_file()

    def test_empty_dialogues(self, generator, tmp_path):
        output_path = tmp_path / "empty.json"
        generator.save_dialogues([], output_path)

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data == []
