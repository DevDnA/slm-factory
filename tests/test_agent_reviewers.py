"""Reviewer 시스템 테스트 — 3 reviewer + aggregator 병렬 실행."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.reviewers import (
    AggregatedVerdict,
    CompletenessChecker,
    GroundingChecker,
    HallucinationChecker,
    ReviewVerdict,
    Reviewer,
    run_reviewers,
)


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_checker(cls, http_client) -> Reviewer:
    return cls(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
    )


class TestEachReviewer:
    """각 reviewer의 기본 동작."""

    @pytest.mark.parametrize("cls, expected_name", [
        (GroundingChecker, "grounding"),
        (CompletenessChecker, "completeness"),
        (HallucinationChecker, "hallucination"),
    ])
    @pytest.mark.asyncio
    async def test_passed_true_정상(self, cls, expected_name):
        raw = '{"passed": true, "reason": "OK"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        v = await _make_checker(cls, http).check("q", "a", [{"doc_id": "d"}])
        assert v.reviewer == expected_name
        assert v.passed is True

    @pytest.mark.parametrize("cls", [
        GroundingChecker, CompletenessChecker, HallucinationChecker,
    ])
    @pytest.mark.asyncio
    async def test_passed_false_with_missing_info(self, cls):
        raw = '{"passed": false, "reason": "부족", "missing_info": "재검색"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        v = await _make_checker(cls, http).check("q", "a", [])
        assert v.passed is False
        assert v.missing_info == "재검색"
        assert v.has_retry_hint is True

    @pytest.mark.asyncio
    async def test_빈_답변은_needs_retry(self):
        http = MagicMock()
        v = await _make_checker(GroundingChecker, http).check("원래질의", "", [])
        assert v.passed is False
        assert v.missing_info == "원래질의"

    @pytest.mark.parametrize("cls", [
        GroundingChecker, CompletenessChecker, HallucinationChecker,
    ])
    @pytest.mark.asyncio
    async def test_HTTP_오류시_pass_처리(self, cls):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("t"))

        v = await _make_checker(cls, http).check("q", "a", [])
        assert v.passed is True  # fallback은 pass
        assert "unavailable" in v.reason

    @pytest.mark.parametrize("cls", [
        GroundingChecker, CompletenessChecker, HallucinationChecker,
    ])
    @pytest.mark.asyncio
    async def test_JSON_파싱_실패시_pass(self, cls):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("not json"))

        v = await _make_checker(cls, http).check("q", "a", [])
        assert v.passed is True
        assert "parse failure" in v.reason


class TestBooleanCoercion:
    @pytest.mark.asyncio
    async def test_문자열_passed_처리(self):
        raw = '{"passed": "yes", "reason": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        v = await _make_checker(GroundingChecker, http).check("q", "a", [])
        assert v.passed is True

    @pytest.mark.asyncio
    async def test_ok_대체_키(self):
        raw = '{"ok": true, "reason": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        v = await _make_checker(GroundingChecker, http).check("q", "a", [])
        assert v.passed is True


class TestAggregator:
    """run_reviewers — 3 reviewer 병렬 실행 + 종합 판정."""

    @pytest.mark.asyncio
    async def test_모두_passed면_overall_passed(self):
        raw = '{"passed": true, "reason": "OK"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        verdict = await run_reviewers(
            query="q",
            answer="a",
            sources=[],
            http_client=http,
            ollama_model="test",
            api_base="http://localhost:11434",
        )
        assert verdict.overall_passed is True
        assert len(verdict.verdicts) == 3
        assert verdict.failed_reviewers == []
        assert not verdict.needs_retry

    @pytest.mark.asyncio
    async def test_하나라도_실패면_overall_false(self):
        # grounding만 false 반환
        class _FakeGrounding(GroundingChecker):
            async def check(self, query, answer, sources):
                return ReviewVerdict(
                    reviewer=self.name,
                    passed=False,
                    reason="근거 약함",
                    missing_info="추가 검색",
                )

        class _FakePass(GroundingChecker):  # 상속을 통해 name만 override할 수 있게
            async def check(self, query, answer, sources):
                return ReviewVerdict(reviewer=self.name, passed=True)

        http = MagicMock()
        gc = _FakeGrounding(http_client=http, ollama_model="t", api_base="x")
        cc_instance = _FakePass(http_client=http, ollama_model="t", api_base="x")
        cc_instance.name = "completeness"
        hc_instance = _FakePass(http_client=http, ollama_model="t", api_base="x")
        hc_instance.name = "hallucination"

        verdict = await run_reviewers(
            query="q",
            answer="a",
            sources=[],
            http_client=http,
            ollama_model="test",
            api_base="http://localhost:11434",
            reviewers=[gc, cc_instance, hc_instance],
        )
        assert verdict.overall_passed is False
        assert "grounding" in verdict.failed_reviewers
        assert verdict.missing_info_query == "추가 검색"
        assert verdict.needs_retry is True

    @pytest.mark.asyncio
    async def test_첫번째_retry_hint_우선_사용(self):
        class _FailingGrounding(GroundingChecker):
            async def check(self, query, answer, sources):
                return ReviewVerdict(
                    reviewer=self.name,
                    passed=False,
                    reason="x",
                    missing_info="grounding 힌트",
                )

        class _FailingCompleteness(GroundingChecker):
            async def check(self, query, answer, sources):
                return ReviewVerdict(
                    reviewer=self.name,
                    passed=False,
                    reason="x",
                    missing_info="completeness 힌트",
                )

        http = MagicMock()
        r1 = _FailingGrounding(http_client=http, ollama_model="t", api_base="x")
        r2 = _FailingCompleteness(http_client=http, ollama_model="t", api_base="x")
        r2.name = "completeness"

        verdict = await run_reviewers(
            query="q",
            answer="a",
            sources=[],
            http_client=http,
            ollama_model="test",
            api_base="x",
            reviewers=[r1, r2],
        )
        # 첫번째 실패 verdict의 hint 사용
        assert verdict.missing_info_query == "grounding 힌트"

    @pytest.mark.asyncio
    async def test_병렬_실행_확인(self):
        """asyncio.gather 사용 — 실행 시간이 합산이 아니라 max."""
        call_log: list[tuple[str, str]] = []

        class _SlowReviewer(GroundingChecker):
            async def check(self, query, answer, sources):
                call_log.append(("start", self.name))
                await asyncio.sleep(0.02)
                call_log.append(("end", self.name))
                return ReviewVerdict(reviewer=self.name, passed=True)

        http = MagicMock()
        r1 = _SlowReviewer(http_client=http, ollama_model="t", api_base="x")
        r1.name = "r1"
        r2 = _SlowReviewer(http_client=http, ollama_model="t", api_base="x")
        r2.name = "r2"
        r3 = _SlowReviewer(http_client=http, ollama_model="t", api_base="x")
        r3.name = "r3"

        await run_reviewers(
            query="q",
            answer="a",
            sources=[],
            http_client=http,
            ollama_model="t",
            api_base="x",
            reviewers=[r1, r2, r3],
        )
        # 셋 다 start가 먼저 찍히고 그 후 end가 찍혀야 병렬 실행
        starts_before_first_end = 0
        for event, _ in call_log:
            if event == "start":
                starts_before_first_end += 1
            elif event == "end":
                break
        assert starts_before_first_end == 3
