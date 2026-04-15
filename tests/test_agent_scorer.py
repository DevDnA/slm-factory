"""Phase 13 — AnswerScorer 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from slm_factory.rag.agent.scorer import AnswerScorer, ScoreResult


def _ollama_response(payload: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"response": payload, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _make_scorer(http_client) -> AnswerScorer:
    return AnswerScorer(
        http_client=http_client,
        ollama_model="test",
        api_base="http://localhost:11434",
        request_timeout=5.0,
    )


class TestValidScoring:
    @pytest.mark.asyncio
    async def test_정상_점수(self):
        raw = '{"score": 8.5, "feedback": "양호", "improvements": ["수치 명시"]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))

        r = await _make_scorer(http).score("질의", "답변", [])
        assert r.score == 8.5
        assert r.feedback == "양호"
        assert r.improvements == ["수치 명시"]

    @pytest.mark.asyncio
    async def test_점수_범위_clamp(self):
        raw = '{"score": 15, "feedback": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))
        r = await _make_scorer(http).score("q", "a", [])
        assert r.score == 10.0

        raw2 = '{"score": -5, "feedback": "x"}'
        http2 = MagicMock()
        http2.post = AsyncMock(return_value=_ollama_response(raw2))
        r2 = await _make_scorer(http2).score("q", "a", [])
        assert r2.score == 1.0

    @pytest.mark.asyncio
    async def test_below_threshold(self):
        raw = '{"score": 5.0, "feedback": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))
        r = await _make_scorer(http).score("q", "a", [])
        assert r.below(7.0)
        assert not r.below(4.0)

    @pytest.mark.asyncio
    async def test_improvements는_빈_항목_제외(self):
        raw = '{"score": 6, "improvements": ["a", "", "b", null]}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))
        r = await _make_scorer(http).score("q", "a", [])
        assert "a" in r.improvements
        assert "b" in r.improvements
        assert "" not in r.improvements


class TestFallback:
    @pytest.mark.asyncio
    async def test_빈_답변은_1점(self):
        http = MagicMock()
        r = await _make_scorer(http).score("q", "", [])
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_HTTP_오류는_중립(self):
        http = MagicMock()
        http.post = AsyncMock(side_effect=httpx.TimeoutException("t"))
        r = await _make_scorer(http).score("q", "a", [])
        assert r.score == 7.0  # neutral
        assert "unavailable" in r.feedback

    @pytest.mark.asyncio
    async def test_JSON_파싱_실패_중립(self):
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response("not json"))
        r = await _make_scorer(http).score("q", "a", [])
        assert r.score == 7.0

    @pytest.mark.asyncio
    async def test_score_문자열은_중립(self):
        raw = '{"score": "high", "feedback": "x"}'
        http = MagicMock()
        http.post = AsyncMock(return_value=_ollama_response(raw))
        r = await _make_scorer(http).score("q", "a", [])
        assert r.score == 7.0


class TestOrchestratorIntegration:
    """Self-improvement loop 통합."""

    @pytest.mark.asyncio
    async def test_low_score면_재합성(self, monkeypatch):
        from tests.test_agent_orchestrator import (
            _PlannerPathFixtures,
            _make_plan,
            _FakeToolResult,
            _collect,
        )
        from slm_factory.rag.agent.orchestrator import AgentOrchestrator
        from slm_factory.rag.agent.router import QueryRouter
        from slm_factory.rag.agent import scorer as scorer_mod
        from types import SimpleNamespace

        # Scorer를 첫 호출 낮은 점수, 두 번째 높은 점수로 mock
        call_count = {"n": 0}

        class _FakeScorer:
            def __init__(self, **_kwargs):
                pass

            async def score(self, query, answer, sources=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return ScoreResult(
                        score=4.0, feedback="부족",
                        improvements=["구체 수치 추가"],
                    )
                return ScoreResult(score=9.0, feedback="개선됨", improvements=[])

        monkeypatch.setattr(scorer_mod, "AnswerScorer", _FakeScorer)

        plan = _make_plan([{"tool": "search", "args": {"query": "q"}}])
        fixtures = _PlannerPathFixtures(
            monkeypatch,
            plan=plan,
            tool_script=[_FakeToolResult(text="r", sources=[])],
            synthesis_scripts=[["초안"], ["개선됨"]],
        )

        config_ns = SimpleNamespace(
            rag=SimpleNamespace(
                agent=SimpleNamespace(
                    enabled=True,
                    max_iterations=3,
                    stream_reasoning=False,
                    planner_enabled=True,
                    verifier_enabled=False,
                    verifier_max_repairs=0,
                    legacy_fallback_enabled=True,
                    session_source_reuse=False,
                    session_source_reuse_limit=5,
                    parallel_steps=False,
                    reflector_enabled=False,
                    reflector_max_retries=1,
                    clarifier_enabled=False,
                    clarifier_max_questions=2,
                    personas_enabled=False,
                    review_work_enabled=False,
                    review_work_retry=False,
                    skills_enabled=False,
                    skills_dir="skills",
                    hooks_enabled=False,
                    builtin_hooks=[],
                    memory_compression_enabled=False,
                    compress_after_turns=10,
                    compress_target_chars=500,
                    self_improvement_enabled=True,
                    min_quality_score=7.0,
                    max_self_improvement_iterations=1,
                    models=SimpleNamespace(
                        router_model="", planner_model="", synthesis_model="",
                        verifier_model="", reviewer_model="", reflector_model="",
                        clarifier_model="", scorer_model="",
                    ),
                ),
                request_timeout=60.0,
            ),
        )

        async def _simple(q):
            yield {"type": "done"}

        orch = AgentOrchestrator(
            router=QueryRouter(agent_enabled=True),
            app_state=fixtures.app_state,
            config=config_ns,
            ollama_model="base",
            api_base="http://localhost:11434",
            rag_max_tokens=-1,
            simple_stream_fn=_simple,
        )
        events = await _collect(orch.handle_agent("질의"))

        tokens = [e["content"] for e in events if e["type"] == "token"]
        # 초안 + 개선됨 둘 다 emit
        assert "초안" in tokens
        assert "개선됨" in tokens
        # synthesis 2회 호출 (초안 + 재합성)
        assert fixtures.app_state.http_client.stream.call_count == 2
