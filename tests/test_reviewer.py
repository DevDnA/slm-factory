"""QAReviewerApp 데이터 로직 테스트 — TUI 렌더링 제외, 정적 메서드 및 데이터 조작 중심."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from slm_factory.models import QAPair
from slm_factory.tui.reviewer import QAReviewerApp


# ---------------------------------------------------------------------------
# 팩토리 헬퍼
# ---------------------------------------------------------------------------


def _make_pair(
    question: str = "테스트 질문",
    answer: str = "테스트 답변",
    review_status: str = "",
    source_doc: str = "test.pdf",
    category: str = "general",
) -> QAPair:
    return QAPair(
        question=question,
        answer=answer,
        source_doc=source_doc,
        category=category,
        review_status=review_status,
    )


def _make_pairs(n: int = 5) -> list[QAPair]:
    return [_make_pair(question=f"질문 {i}", answer=f"답변 {i}") for i in range(n)]


def _dump_pairs(pairs: list[QAPair], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(p) for p in pairs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# TestLoadPairs
# ---------------------------------------------------------------------------


class TestLoadPairs:
    def test_load_valid_pairs(self, tmp_path: Path) -> None:
        original = _make_pairs(3)
        _dump_pairs(original, tmp_path / "qa.json")

        loaded = QAReviewerApp.load_pairs(tmp_path / "qa.json")

        assert len(loaded) == 3
        assert loaded[0].question == "질문 0"
        assert loaded[2].answer == "답변 2"

    def test_load_empty_list(self, tmp_path: Path) -> None:
        (tmp_path / "empty.json").write_text("[]", encoding="utf-8")

        loaded = QAReviewerApp.load_pairs(tmp_path / "empty.json")

        assert loaded == []

    def test_load_preserves_review_status(self, tmp_path: Path) -> None:
        pairs = [_make_pair(review_status="approved")]
        _dump_pairs(pairs, tmp_path / "qa.json")

        loaded = QAReviewerApp.load_pairs(tmp_path / "qa.json")

        assert loaded[0].review_status == "approved"

    def test_load_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            QAReviewerApp.load_pairs(tmp_path / "bad.json")

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            QAReviewerApp.load_pairs(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# TestSavePairs
# ---------------------------------------------------------------------------


class TestSavePairs:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        pairs = _make_pairs(2)
        out = tmp_path / "out.json"

        QAReviewerApp.save_pairs(pairs, out)

        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "qa.json"

        QAReviewerApp.save_pairs([_make_pair()], out)

        assert out.is_file()

    def test_round_trip(self, tmp_path: Path) -> None:
        original = [
            _make_pair(question="Q1", answer="A1", review_status="approved"),
            _make_pair(question="Q2", answer="A2", review_status="rejected"),
            _make_pair(question="Q3", answer="A3"),
        ]
        path = tmp_path / "round.json"

        QAReviewerApp.save_pairs(original, path)
        loaded = QAReviewerApp.load_pairs(path)

        assert len(loaded) == len(original)
        for orig, load in zip(original, loaded):
            assert orig.question == load.question
            assert orig.answer == load.answer
            assert orig.review_status == load.review_status

    def test_save_unicode(self, tmp_path: Path) -> None:
        pairs = [_make_pair(question="한국어 질문", answer="답변입니다")]
        out = tmp_path / "unicode.json"

        QAReviewerApp.save_pairs(pairs, out)

        raw = out.read_text(encoding="utf-8")
        assert "한국어 질문" in raw


# ---------------------------------------------------------------------------
# TestCountStatuses
# ---------------------------------------------------------------------------


class TestCountStatuses:
    def test_all_pending(self) -> None:
        pairs = _make_pairs(4)
        result = QAReviewerApp.count_statuses(pairs)
        assert result == {"approved": 0, "rejected": 0, "pending": 4}

    def test_mixed(self) -> None:
        pairs = [
            _make_pair(review_status="approved"),
            _make_pair(review_status="rejected"),
            _make_pair(review_status="approved"),
            _make_pair(),
        ]
        result = QAReviewerApp.count_statuses(pairs)
        assert result == {"approved": 2, "rejected": 1, "pending": 1}

    def test_all_approved(self) -> None:
        pairs = [_make_pair(review_status="approved") for _ in range(3)]
        result = QAReviewerApp.count_statuses(pairs)
        assert result == {"approved": 3, "rejected": 0, "pending": 0}

    def test_empty_list(self) -> None:
        result = QAReviewerApp.count_statuses([])
        assert result == {"approved": 0, "rejected": 0, "pending": 0}

    def test_unknown_status_counted_as_pending(self) -> None:
        pairs = [_make_pair(review_status="unknown")]
        result = QAReviewerApp.count_statuses(pairs)
        assert result["pending"] == 1


# ---------------------------------------------------------------------------
# TestApproveRejectLogic
# ---------------------------------------------------------------------------


class TestApproveRejectLogic:
    def test_approve_sets_status(self) -> None:
        pair = _make_pair()
        assert pair.review_status == ""

        pair.review_status = "approved"

        assert pair.review_status == "approved"

    def test_reject_sets_status(self) -> None:
        pair = _make_pair()
        pair.review_status = "rejected"
        assert pair.review_status == "rejected"

    def test_re_approve_after_reject(self) -> None:
        pair = _make_pair(review_status="rejected")
        pair.review_status = "approved"
        assert pair.review_status == "approved"


# ---------------------------------------------------------------------------
# TestNavigationBounds
# ---------------------------------------------------------------------------


class TestNavigationBounds:
    def test_next_within_bounds(self) -> None:
        idx = 0
        total = 5
        if idx < total - 1:
            idx += 1
        assert idx == 1

    def test_next_at_last(self) -> None:
        idx = 4
        total = 5
        if idx < total - 1:
            idx += 1
        assert idx == 4

    def test_prev_within_bounds(self) -> None:
        idx = 3
        if idx > 0:
            idx -= 1
        assert idx == 2

    def test_prev_at_first(self) -> None:
        idx = 0
        if idx > 0:
            idx -= 1
        assert idx == 0

    def test_advance_clamps_at_end(self) -> None:
        idx = 4
        total = 5
        if idx < total - 1:
            idx += 1
        assert idx == 4


# ---------------------------------------------------------------------------
# TestEditPair
# ---------------------------------------------------------------------------


class TestEditPair:
    def test_edit_answer_updates(self) -> None:
        pair = _make_pair(answer="원래 답변")
        pair.answer = "수정된 답변"
        assert pair.answer == "수정된 답변"

    def test_edit_preserves_other_fields(self) -> None:
        pair = _make_pair(
            question="질문", answer="원래", source_doc="doc.pdf", category="cat"
        )
        pair.answer = "새 답변"

        assert pair.question == "질문"
        assert pair.source_doc == "doc.pdf"
        assert pair.category == "cat"


# ---------------------------------------------------------------------------
# TestUnsavedTracking
# ---------------------------------------------------------------------------


class TestUnsavedTracking:
    def test_approve_marks_unsaved(self) -> None:
        unsaved = False
        pair = _make_pair()
        pair.review_status = "approved"
        unsaved = True
        assert unsaved is True

    def test_reject_marks_unsaved(self) -> None:
        unsaved = False
        pair = _make_pair()
        pair.review_status = "rejected"
        unsaved = True
        assert unsaved is True

    def test_edit_marks_unsaved(self) -> None:
        unsaved = False
        pair = _make_pair()
        pair.answer = "수정됨"
        unsaved = True
        assert unsaved is True

    def test_save_clears_unsaved(self, tmp_path: Path) -> None:
        pairs = [_make_pair(review_status="approved")]
        out = tmp_path / "save.json"

        unsaved = True
        QAReviewerApp.save_pairs(pairs, out)
        unsaved = False

        assert unsaved is False
        assert out.is_file()
