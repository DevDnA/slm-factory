"""Reflector 테스트 — 답변 품질 판정, JSON 파싱, fallback 동작."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.reflector import Reflector, ReflectorDecision


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_reflector(http_client) -> Reflector:
    return Reflector(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
    )


class TestValidDecision:
    """구조화된 JSON 응답."""

    @pytest.mark.asyncio
    async def test_answer_ok_true(self):
        raw = '{"answer_ok": true, "reason": "근거 명확함"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_reflector(http).reflect("질문", "답변", [])
        assert d.answer_ok is True
        assert "근거 명확" in d.reason
        assert not d.needs_retry

    @pytest.mark.asyncio
    async def test_answer_ok_false_with_missing(self):
        raw = (
            '{"answer_ok": false, "reason": "예외 조항 누락", '
            '"missing_info_query": "예외 조항"}'
        )
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_reflector(http).reflect("질문", "답변", [])
        assert d.answer_ok is False
        assert d.missing_info_query == "예외 조항"
        assert d.needs_retry

    @pytest.mark.asyncio
    async def test_missing_info_대체키(self):
        raw = '{"answer_ok": false, "reason": "x", "next_query": "추가"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.missing_info_query == "추가"

    @pytest.mark.asyncio
    async def test_없는_키_기본_True(self):
        raw = '{"reason": "no answer_ok key"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.answer_ok is True

    @pytest.mark.asyncio
    async def test_빈_문자열_missing_정규화(self):
        raw = '{"answer_ok": false, "reason": "x", "missing_info_query": ""}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.missing_info_query is None
        assert not d.needs_retry


class TestBooleanCoercion:
    @pytest.mark.parametrize("raw_val, expected", [
        ('"true"', True),
        ('"yes"', True),
        ('"예"', True),
        ('"ok"', True),
        ('"좋음"', True),
        ('"false"', False),
        ('"no"', False),
    ])
    @pytest.mark.asyncio
    async def test_문자열_answer_ok(self, raw_val, expected):
        raw = f'{{"answer_ok": {raw_val}, "reason": "x"}}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.answer_ok is expected


class TestFallback:
    """reflect()는 어떤 실패에서도 raise하지 않고 answer_ok=True로 반환."""

    @pytest.mark.asyncio
    async def test_빈_답변은_needs_retry_True(self):
        http = MagicMock()
        d = await _make_reflector(http).reflect("질문", "", [])
        assert not d.answer_ok
        assert d.needs_retry
        # LLM 호출 없이 즉시 반환
        http.post.assert_not_called() if hasattr(http.post, "assert_not_called") else None

    @pytest.mark.asyncio
    async def test_HTTP_타임아웃시_answer_ok_True(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.answer_ok is True
        assert "unavailable" in d.reason

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패시_answer_ok_True(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("not json"))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.answer_ok is True

    @pytest.mark.asyncio
    async def test_JSON_배열은_dict_아님_fallback(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("[false]"))

        d = await _make_reflector(http).reflect("q", "a", [])
        assert d.answer_ok is True


class TestSourcesFormatting:
    """프롬프트에 sources가 올바르게 요약되어 전달되는지."""

    @pytest.mark.asyncio
    async def test_sources_프리뷰_전달(self):
        raw = '{"answer_ok": true, "reason": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        sources = [
            {"doc_id": "d1", "content": "첫번째 문서 내용"},
            {"doc_id": "d2", "content": "두번째 문서 내용"},
        ]
        await _make_reflector(http).reflect("질문", "답변", sources)

        call_kwargs = http.post.call_args.kwargs
        prompt = call_kwargs["json"]["prompt"]
        assert "d1" in prompt
        assert "d2" in prompt
        assert "첫번째 문서 내용" in prompt

    @pytest.mark.asyncio
    async def test_빈_sources(self):
        raw = '{"answer_ok": true, "reason": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        await _make_reflector(http).reflect("질문", "답변", [])

        call_kwargs = http.post.call_args.kwargs
        prompt = call_kwargs["json"]["prompt"]
        assert "(수집된 참고 문서 없음)" in prompt


class TestReflectorDecision:
    def test_needs_retry_true(self):
        d = ReflectorDecision(answer_ok=False, missing_info_query="new")
        assert d.needs_retry

    def test_needs_retry_ok면_false(self):
        d = ReflectorDecision(answer_ok=True, missing_info_query="ignored")
        assert not d.needs_retry

    def test_needs_retry_missing_없으면_false(self):
        d = ReflectorDecision(answer_ok=False, missing_info_query=None)
        assert not d.needs_retry
