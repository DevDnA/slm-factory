"""`/auto` 엔드포인트를 위한 라우팅·스트리밍 오케스트레이터.

``AgentOrchestrator``는 질의를 받아 다음을 수행합니다.

1. ``QueryRouter``로 ``simple`` | ``agent`` 경로 결정
2. ``{"type": "route"}`` 이벤트 발행
3. 선택된 경로의 이벤트 스트림을 그대로 전달

이벤트는 dict 형태로 yield되며, ``server.py``는 SSE로 framing만 담당합니다.
이렇게 분리하면 라우팅·세션 관리·agent 이벤트 매핑 로직을 HTTP 레이어
없이 단독으로 테스트할 수 있습니다.

동작 보존 원칙
--------------
기존 ``server.py``의 ``/auto`` 핸들러가 발행하던 이벤트 순서와 필드는
**바이트 수준으로 동일하게** 유지됩니다. 본 모듈은 순수 추출 리팩터링이며
새로운 기능(planner, verifier 등)은 포함하지 않습니다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Callable

from ...utils import get_logger
from .persona_router import PersonaRouter
from .personas.base import Persona
from .router import QueryRouter

logger = get_logger("rag.agent.orchestrator")


SimpleStreamFn = Callable[[str], AsyncGenerator[dict, None]]
"""단순 RAG 스트림 함수 — query를 받아 이벤트 dict를 yield합니다."""

# 컨텍스트 합성 시 prompt에 삽입되는 참고 문서의 최대 길이.
_SYNTHESIS_CONTEXT_CHAR_LIMIT = 6000
# observation 이벤트로 클라이언트에 보낼 때의 길이 제한.
_OBSERVATION_PREVIEW_LIMIT = 300


class AgentOrchestrator:
    """``/auto`` 경로의 라우팅·스트리밍 오케스트레이터.

    Parameters
    ----------
    router:
        복잡도 기반 라우팅 결정기.
    app_state:
        FastAPI ``app.state`` — 런타임에 ``agent_session_manager``,
        ``agent_tool_registry``, ``http_client``를 조회합니다.
    config:
        ``SLMConfig`` — ``rag.agent``, ``rag.request_timeout``,
        ``rag.max_tokens`` 등을 참조합니다.
    ollama_model:
        Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    rag_max_tokens:
        LLM 생성 최대 토큰.
    simple_stream_fn:
        단순 RAG 스트림을 생성하는 async generator factory — ``app.state``가
        보유한 Qdrant·임베딩·reranker 등 의존성을 클로저로 캡처한 함수를
        주입받습니다.
    """

    def __init__(
        self,
        *,
        router: QueryRouter,
        app_state: Any,
        config: Any,
        ollama_model: str,
        api_base: str,
        rag_max_tokens: int,
        simple_stream_fn: SimpleStreamFn,
    ) -> None:
        self._router = router
        self._app_state = app_state
        self._config = config
        self._ollama_model = ollama_model
        self._api_base = api_base
        self._rag_max_tokens = rag_max_tokens
        self._simple_stream_fn = simple_stream_fn
        self._persona_router = PersonaRouter(
            enabled=getattr(config.rag.agent, "personas_enabled", False),
            custom_registry=self._build_custom_personas(config),
        )
        self._skill_registry = self._build_skill_registry(config)
        self._hook_registry = self._build_hook_registry(config)

    @staticmethod
    def _build_hook_registry(config: Any):
        """config.builtin_hooks로 지정된 hook들을 등록한 registry 반환."""
        from .hooks import build_default_registry

        enabled = getattr(config.rag.agent, "hooks_enabled", False)
        names = list(getattr(config.rag.agent, "builtin_hooks", []) or [])
        return build_default_registry(enabled=enabled, builtin_names=names)

    def register_hook(self, point: str, fn):
        """외부 코드가 orchestrator에 사용자 정의 hook을 등록할 수 있도록 제공."""
        self._hook_registry.register(point, fn)

    def _model_for(self, slot: str) -> str:
        """Phase 9 — 컴포넌트별 모델 슬롯 조회. 빈 값이면 기본 모델로 fallback."""
        models_cfg = getattr(self._config.rag.agent, "models", None)
        if models_cfg is None:
            return self._ollama_model
        value = getattr(models_cfg, f"{slot}_model", "") or ""
        return value.strip() or self._ollama_model

    @staticmethod
    def _build_custom_personas(config: Any):
        """Phase 14 — custom_personas_dir가 설정되면 YAML에서 로드."""
        from .persona_loader import CustomPersonaRegistry, load_custom_personas

        path = (getattr(config.rag.agent, "custom_personas_dir", "") or "").strip()
        if not path:
            return None
        try:
            personas = load_custom_personas(path)
        except Exception as exc:  # pragma: no cover — loader는 자체 never-raise
            logger.warning("Custom personas 로드 실패: %s", exc)
            personas = []
        if personas:
            logger.info(
                "Custom personas 로드: %d개 (%s)",
                len(personas),
                ", ".join(p.name for p in personas),
            )
        return CustomPersonaRegistry(personas)

    @staticmethod
    def _build_skill_registry(config: Any):
        """Skills 디렉터리에서 Skill 목록을 로드 — 실패 시 빈 registry."""
        from .skills import SkillRegistry, load_skills_from_dir

        if not getattr(config.rag.agent, "skills_enabled", False):
            return SkillRegistry()
        skills_dir = getattr(config.rag.agent, "skills_dir", "skills")
        try:
            skills = load_skills_from_dir(skills_dir)
        except Exception as exc:  # pragma: no cover — loader는 never-raise
            logger.warning("Skills 로드 실패: %s — 빈 registry 사용", exc)
            skills = []
        if skills:
            logger.info(
                "Skills 로드: %d개 (%s)",
                len(skills),
                ", ".join(s.name for s in skills),
            )
        return SkillRegistry(skills)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_auto(
        self, query: str, session_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """``/auto``의 전체 이벤트 스트림을 생성합니다."""
        query = await self._hook_registry.run("pre_query", query)
        # IntentClassifier가 주입된 router는 ``route_async()``를 통해 LLM 분류를 수행.
        decision = await self._router.route_async(query)
        logger.info(
            "라우팅 결정: mode=%s complexity=%.2f reason=%s intent=%s",
            decision.mode,
            decision.complexity,
            decision.reason,
            decision.intent,
        )

        route_event: dict[str, Any] = {"type": "route", "mode": decision.mode}
        if decision.intent is not None:
            route_event["intent"] = decision.intent
        yield route_event

        # Clarifier: ambiguous 의도 + clarifier 활성화 시 명확화 질문 반환.
        if (
            decision.intent == "ambiguous"
            and self._config.rag.agent.clarifier_enabled
        ):
            async for event in self._stream_clarifier(query, session_id):
                yield event
            return

        persona = self._persona_router.select(decision.intent)
        if persona is not None:
            logger.info("Persona 선택: %s", persona.name)

        if decision.mode == "simple":
            async for event in self._simple_stream_fn(query):
                yield event
        else:
            async for event in self._stream_agent(query, session_id, persona=persona):
                yield event

    async def handle_agent(
        self, query: str, session_id: str | None = None
    ) -> AsyncGenerator[dict, None]:
        """``/agent`` stream 모드 — 라우팅 없이 항상 agent 경로.

        ``handle_auto``와 달리 ``{type: route}`` 이벤트를 발행하지 않으며,
        ``planner_enabled`` 설정에 따라 planner 또는 legacy 경로로 분기합니다.
        """
        query = await self._hook_registry.run("pre_query", query)
        async for event in self._stream_agent(query, session_id):
            yield event

    # ------------------------------------------------------------------
    # 내부: Clarifier 경로 — ambiguous 의도에 대한 역질문
    # ------------------------------------------------------------------

    async def _stream_clarifier(
        self, query: str, session_id: str | None
    ) -> AsyncGenerator[dict, None]:
        """Clarifier persona로 명확화 질문을 생성·반환합니다."""
        from .personas.clarifier import Clarifier
        from .session import Message

        session_store = self._app_state.agent_session_manager
        http_client = self._app_state.http_client

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)
        session_store.add_message(sid, Message(role="user", content=query))

        clarifier = Clarifier(
            http_client=http_client,
            ollama_model=self._model_for("clarifier"),
            api_base=self._api_base,
            request_timeout=min(self._config.rag.request_timeout, 15.0),
            max_questions=self._config.rag.agent.clarifier_max_questions,
        )
        result = await clarifier.generate_questions(query, history=history)

        # 세션에 assistant 턴으로 기록 — 다음 턴에 이전 역질문 맥락을 이어감.
        summary = "명확화 질문: " + " / ".join(result.questions)
        session_store.add_message(sid, Message(role="assistant", content=summary))

        yield {
            "type": "clarification",
            "questions": result.questions,
            "is_fallback": result.metadata.get("is_fallback", False),
        }
        yield {"type": "done", "session_id": sid}

    # ------------------------------------------------------------------
    # 내부: agent 경로 디스패치
    # ------------------------------------------------------------------

    async def _stream_agent(
        self,
        query: str,
        session_id: str | None,
        *,
        persona: Persona | None = None,
    ) -> AsyncGenerator[dict, None]:
        """``planner_enabled`` 설정에 따라 planner 또는 legacy 경로로 디스패치합니다."""
        if self._config.rag.agent.planner_enabled:
            async for ev in self._stream_agent_planner(
                query, session_id, persona=persona
            ):
                yield ev
        else:
            async for ev in self._stream_agent_legacy(query, session_id):
                yield ev

    # ------------------------------------------------------------------
    # 내부: legacy agent 경로 — 기존 ReAct AgentLoop
    # ------------------------------------------------------------------

    async def _stream_agent_legacy(
        self, query: str, session_id: str | None
    ) -> AsyncGenerator[dict, None]:
        """Agent RAG legacy 경로 — 세션 관리 + AgentLoop run_stream 이벤트 매핑."""
        from .loop import AgentLoop
        from .session import Message

        session_store = self._app_state.agent_session_manager
        tool_registry = self._app_state.agent_tool_registry
        http_client = self._app_state.http_client

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)

        agent = AgentLoop(
            http_client=http_client,
            tool_registry=tool_registry,
            ollama_model=self._model_for("synthesis"),
            api_base=self._api_base,
            max_iterations=self._config.rag.agent.max_iterations,
            max_tokens=self._rag_max_tokens,
            request_timeout=self._config.rag.request_timeout,
        )

        session_store.add_message(sid, Message(role="user", content=query))

        stream_reasoning = self._config.rag.agent.stream_reasoning
        answer_parts: list[str] = []
        final_sources: list[dict] = []

        try:
            async for event in agent.run_stream(query, history):
                if event.type == "thought" and stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": event.content,
                        "iteration": event.iteration,
                    }
                elif event.type == "action" and stream_reasoning:
                    yield {
                        "type": "action",
                        "content": event.content,
                        "input": event.metadata.get("input", {}),
                        "iteration": event.iteration,
                    }
                elif event.type == "observation" and stream_reasoning:
                    obs_preview = event.content[:300]
                    if len(event.content) > 300:
                        obs_preview += "..."
                    yield {
                        "type": "observation",
                        "content": obs_preview,
                        "iteration": event.iteration,
                    }
                elif event.type == "token":
                    answer_parts.append(event.content)
                    yield {"type": "token", "content": event.content}
                elif event.type == "done":
                    final_sources = event.metadata.get("sources", [])
                elif event.type == "error":
                    yield {
                        "type": "token",
                        "content": "[오류] 처리 중 문제가 발생했습니다.",
                    }
        except Exception as exc:
            logger.error("Agent 스트리밍 오류: %s", exc)
            yield {
                "type": "token",
                "content": "[오류] 처리 중 문제가 발생했습니다.",
            }

        answer = "".join(answer_parts)
        if answer:
            session_store.add_message(sid, Message(role="assistant", content=answer))

        sources_payload = [
            {
                "content": s.get("content", ""),
                "doc_id": s.get("doc_id", ""),
                "score": s.get("score", 0.0),
            }
            for s in final_sources
        ]
        yield {"type": "sources", "sources": sources_payload}
        yield {"type": "done", "session_id": sid}

    # ------------------------------------------------------------------
    # 내부: planner 경로 — plan → execute → verify → synthesize
    # ------------------------------------------------------------------

    async def _stream_agent_planner(
        self,
        query: str,
        session_id: str | None,
        *,
        persona: Persona | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Planner 기반 오케스트레이션 경로."""
        from .planner import Planner
        from .session import Message
        from .verifier import Verifier

        http_client = self._app_state.http_client

        stream_reasoning = self._config.rag.agent.stream_reasoning
        # Planner/Verifier는 메인 timeout보다 짧게 — 빠른 실패로 fallback 경로 확보.
        aux_timeout = min(self._config.rag.request_timeout, 30.0)

        # Plan 먼저 생성 — 이 시점까지는 세션에 부수효과 없음.
        planner = Planner(
            http_client=http_client,
            ollama_model=self._model_for("planner"),
            api_base=self._api_base,
            request_timeout=aux_timeout,
        )
        plan = await planner.plan(query)

        # Persona가 도구 권한을 제한하면 plan step을 필터링.
        if persona is not None and persona.allowed_tools is not None:
            allowed = persona.allowed_tools
            if allowed:
                original_count = len(plan.steps)
                plan.steps = [s for s in plan.steps if s.tool in allowed]
                if len(plan.steps) < original_count:
                    logger.debug(
                        "Persona '%s' 도구 화이트리스트로 step %d → %d",
                        persona.name,
                        original_count,
                        len(plan.steps),
                    )
            # 빈 allowed_tools는 "도구 없음"이므로 plan을 비움.
            else:
                plan.steps = []

        # Fallback 게이트 — planner가 구조적으로 실패했으면 legacy 경로로 위임.
        # 세션 user 메시지는 legacy가 기록하므로 여기서는 건너뜁니다.
        if plan.is_fallback and self._config.rag.agent.legacy_fallback_enabled:
            logger.warning(
                "Planner fallback (%s) — legacy AgentLoop 경로로 전환",
                plan.rationale,
            )
            async for event in self._stream_agent_legacy(query, session_id):
                yield event
            return

        session_store = self._app_state.agent_session_manager
        tool_registry = self._app_state.agent_tool_registry

        sid, _ = session_store.get_or_create(session_id)
        history = session_store.format_history(sid)
        session_store.add_message(sid, Message(role="user", content=query))

        if stream_reasoning:
            summary = f"계획: {plan.strategy} 전략, {len(plan.steps)}개 step"
            if plan.rationale:
                summary += f" — {plan.rationale}"
            yield {"type": "thought", "content": summary, "iteration": 0}

        all_sources: list[dict] = []
        context_parts: list[str] = []

        # --- Plan step 실행 -------------------------------------------
        # 병렬 조건: parallel_steps=True + search-only + 2개 이상.
        can_parallelize = (
            self._config.rag.agent.parallel_steps
            and len(plan.steps) >= 2
            and all(step.tool == "search" for step in plan.steps)
        )

        if can_parallelize:
            # 동시 실행 후 결과를 plan 순서대로 이벤트 발행.
            try:
                results = await asyncio.gather(
                    *[
                        tool_registry.execute(step.tool, step.args)
                        for step in plan.steps
                    ],
                    return_exceptions=True,
                )
            except Exception as exc:
                logger.warning("병렬 step 실행 실패: %s", exc)
                results = [exc] * len(plan.steps)

            for i, (step, result) in enumerate(zip(plan.steps, results), start=1):
                if stream_reasoning and step.reason:
                    yield {
                        "type": "thought",
                        "content": step.reason,
                        "iteration": i,
                    }
                if stream_reasoning:
                    yield {
                        "type": "action",
                        "content": step.tool,
                        "input": step.args,
                        "iteration": i,
                    }
                if isinstance(result, Exception):
                    logger.warning(
                        "병렬 step '%s' 실패: %s — 건너뜁니다", step.tool, result
                    )
                    continue

                for src in result.sources:
                    if not any(
                        s.get("doc_id") == src.get("doc_id") for s in all_sources
                    ):
                        all_sources.append(src)
                context_parts.append(result.text)

                if stream_reasoning:
                    obs_preview = result.text[:_OBSERVATION_PREVIEW_LIMIT]
                    if len(result.text) > _OBSERVATION_PREVIEW_LIMIT:
                        obs_preview += "..."
                    yield {
                        "type": "observation",
                        "content": obs_preview,
                        "iteration": i,
                    }
        else:
            # 직렬 실행 — 기본 경로. 도구 간 의존성이 있을 수 있어 안전.
            for i, step in enumerate(plan.steps, start=1):
                if stream_reasoning and step.reason:
                    yield {
                        "type": "thought",
                        "content": step.reason,
                        "iteration": i,
                    }
                if stream_reasoning:
                    yield {
                        "type": "action",
                        "content": step.tool,
                        "input": step.args,
                        "iteration": i,
                    }

                try:
                    result = await tool_registry.execute(step.tool, step.args)
                except Exception as exc:
                    logger.warning("도구 '%s' 실행 실패: %s", step.tool, exc)
                    continue

                for src in result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)
                context_parts.append(result.text)

                if stream_reasoning:
                    obs_preview = result.text[:_OBSERVATION_PREVIEW_LIMIT]
                    if len(result.text) > _OBSERVATION_PREVIEW_LIMIT:
                        obs_preview += "..."
                    yield {
                        "type": "observation",
                        "content": obs_preview,
                        "iteration": i,
                    }

        # --- Verifier 기반 repair 루프 ---------------------------------
        repair_iteration = len(plan.steps)
        if self._config.rag.agent.verifier_enabled:
            verifier = Verifier(
                http_client=http_client,
                ollama_model=self._model_for("verifier"),
                api_base=self._api_base,
                request_timeout=aux_timeout,
            )
            max_repairs = self._config.rag.agent.verifier_max_repairs
            for _ in range(max_repairs):
                context_str = "\n\n".join(context_parts)
                decision = await verifier.evaluate(query, context_str)
                if not decision.needs_repair:
                    break

                repair_iteration += 1
                suggested = decision.suggested_query or ""

                if stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": (
                            f"추가 검색 필요: {decision.reason} → '{suggested}'"
                        ),
                        "iteration": repair_iteration,
                    }
                    yield {
                        "type": "action",
                        "content": "search",
                        "input": {"query": suggested},
                        "iteration": repair_iteration,
                    }

                try:
                    repair_result = await tool_registry.execute(
                        "search", {"query": suggested}
                    )
                except Exception as exc:
                    logger.warning("Repair search 실패: %s", exc)
                    break

                for src in repair_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)
                context_parts.append(repair_result.text)

                if stream_reasoning:
                    obs_preview = repair_result.text[:_OBSERVATION_PREVIEW_LIMIT]
                    if len(repair_result.text) > _OBSERVATION_PREVIEW_LIMIT:
                        obs_preview += "..."
                    yield {
                        "type": "observation",
                        "content": obs_preview,
                        "iteration": repair_iteration,
                    }

        # --- 답변 합성 -------------------------------------------------
        # 이전 턴의 참조 문서를 synthesis 컨텍스트에 주입(follow-up 연속성).
        prior_context = self._format_prior_context(session_store, sid)
        context_str = "\n\n".join(context_parts)
        if prior_context:
            context_str = f"{prior_context}\n\n{context_str}" if context_str else prior_context

        # Skills 주입: 질의에 매칭되는 skill의 prompt_addon을 context 앞에 prepend.
        skill_addon = self._format_active_skills(query)
        if skill_addon:
            context_str = f"{skill_addon}\n\n{context_str}" if context_str else skill_addon

        # post_search hook — 수집된 source 목록을 후처리(dedup, boosting 등).
        all_sources = await self._hook_registry.run("post_search", all_sources)

        answer_parts: list[str] = []
        async for token in self._stream_synthesis(
            query, context_str, history, persona=persona
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
        answer = "".join(answer_parts)
        answer = await self._hook_registry.run("post_synthesis", answer)

        # --- Reflector: 답변 자기 검증 + 필요 시 추가 검색·재합성 -------
        if self._config.rag.agent.reflector_enabled and answer:
            from .reflector import Reflector

            reflector = Reflector(
                http_client=http_client,
                ollama_model=self._model_for("reflector"),
                api_base=self._api_base,
                request_timeout=aux_timeout,
            )
            max_retries = self._config.rag.agent.reflector_max_retries
            reflect_iteration = repair_iteration
            for _ in range(max_retries):
                current_sources = [
                    {
                        "content": s.get("content", ""),
                        "doc_id": s.get("doc_id", ""),
                        "score": s.get("score", 0.0),
                    }
                    for s in all_sources
                ]
                decision = await reflector.reflect(query, answer, current_sources)
                if not decision.needs_retry:
                    break

                reflect_iteration += 1
                missing_q = decision.missing_info_query or ""

                if stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": (
                            f"답변 자기 검증: {decision.reason} → 보완 검색 '{missing_q}'"
                        ),
                        "iteration": reflect_iteration,
                    }
                    yield {
                        "type": "action",
                        "content": "search",
                        "input": {"query": missing_q},
                        "iteration": reflect_iteration,
                    }

                try:
                    extra = await tool_registry.execute("search", {"query": missing_q})
                except Exception as exc:
                    logger.warning("Reflector 보완 검색 실패: %s", exc)
                    break

                for src in extra.sources:
                    if not any(
                        s.get("doc_id") == src.get("doc_id") for s in all_sources
                    ):
                        all_sources.append(src)
                context_parts.append(extra.text)

                if stream_reasoning:
                    obs_preview = extra.text[:_OBSERVATION_PREVIEW_LIMIT]
                    if len(extra.text) > _OBSERVATION_PREVIEW_LIMIT:
                        obs_preview += "..."
                    yield {
                        "type": "observation",
                        "content": obs_preview,
                        "iteration": reflect_iteration,
                    }
                    yield {
                        "type": "thought",
                        "content": "보완된 컨텍스트로 답변을 재생성합니다.",
                        "iteration": reflect_iteration,
                    }

                new_context = "\n\n".join(context_parts)
                if prior_context:
                    new_context = (
                        f"{prior_context}\n\n{new_context}" if new_context else prior_context
                    )
                retry_parts: list[str] = []
                async for token in self._stream_synthesis(
                    query, new_context, history, persona=persona
                ):
                    retry_parts.append(token)
                    yield {"type": "token", "content": token}
                retry_answer = "".join(retry_parts)
                if retry_answer:
                    answer = retry_answer

        # --- Review-Work: 3 reviewer 병렬 검증 + 선택적 재합성 -------------
        if self._config.rag.agent.review_work_enabled and answer:
            current_sources = [
                {
                    "content": s.get("content", ""),
                    "doc_id": s.get("doc_id", ""),
                    "score": s.get("score", 0.0),
                }
                for s in all_sources
            ]
            from .reviewers import run_reviewers

            verdict = await run_reviewers(
                query=query,
                answer=answer,
                sources=current_sources,
                http_client=http_client,
                ollama_model=self._model_for("reviewer"),
                api_base=self._api_base,
                request_timeout=aux_timeout,
            )
            for v in verdict.verdicts:
                yield {
                    "type": "review",
                    "reviewer": v.reviewer,
                    "passed": v.passed,
                    "reason": v.reason,
                }

            # 선택적 자동 재시도: review 실패 + missing_info 있음 + 재시도 허용
            if (
                self._config.rag.agent.review_work_retry
                and verdict.needs_retry
            ):
                repair_iteration += 1
                rq = verdict.missing_info_query or ""
                if stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": (
                            f"Review-Work 재시도: failed={verdict.failed_reviewers} "
                            f"→ '{rq}'"
                        ),
                        "iteration": repair_iteration,
                    }
                    yield {
                        "type": "action",
                        "content": "search",
                        "input": {"query": rq},
                        "iteration": repair_iteration,
                    }
                try:
                    review_extra = await tool_registry.execute(
                        "search", {"query": rq}
                    )
                except Exception as exc:
                    logger.warning("Review-Work 재시도 검색 실패: %s", exc)
                    review_extra = None

                if review_extra is not None:
                    for src in review_extra.sources:
                        if not any(
                            s.get("doc_id") == src.get("doc_id") for s in all_sources
                        ):
                            all_sources.append(src)
                    context_parts.append(review_extra.text)

                    if stream_reasoning:
                        obs_preview = review_extra.text[:_OBSERVATION_PREVIEW_LIMIT]
                        if len(review_extra.text) > _OBSERVATION_PREVIEW_LIMIT:
                            obs_preview += "..."
                        yield {
                            "type": "observation",
                            "content": obs_preview,
                            "iteration": repair_iteration,
                        }

                    new_context = "\n\n".join(context_parts)
                    if prior_context:
                        new_context = (
                            f"{prior_context}\n\n{new_context}"
                            if new_context
                            else prior_context
                        )
                    review_retry_parts: list[str] = []
                    async for token in self._stream_synthesis(
                        query, new_context, history, persona=persona
                    ):
                        review_retry_parts.append(token)
                        yield {"type": "token", "content": token}
                    review_retry_answer = "".join(review_retry_parts)
                    if review_retry_answer:
                        answer = review_retry_answer

        # --- Phase 13: Self-Improvement 점수 기반 재시도 ---------------
        if getattr(self._config.rag.agent, "self_improvement_enabled", False) and answer:
            from .scorer import AnswerScorer

            scorer = AnswerScorer(
                http_client=http_client,
                ollama_model=self._model_for("scorer"),
                api_base=self._api_base,
                request_timeout=aux_timeout,
            )
            min_score = getattr(self._config.rag.agent, "min_quality_score", 7.0)
            max_iters = getattr(
                self._config.rag.agent, "max_self_improvement_iterations", 1
            )
            for _ in range(max_iters):
                current_sources_for_score = [
                    {
                        "content": s.get("content", ""),
                        "doc_id": s.get("doc_id", ""),
                        "score": s.get("score", 0.0),
                    }
                    for s in all_sources
                ]
                result = await scorer.score(
                    query, answer, current_sources_for_score
                )
                if not result.below(min_score):
                    break

                if stream_reasoning:
                    yield {
                        "type": "thought",
                        "content": (
                            f"자기 개선: 점수 {result.score:.1f}/10 "
                            f"< {min_score} → 재합성 시도"
                        ),
                        "iteration": 0,
                    }

                # Feedback을 context 앞에 주입하고 재합성.
                feedback_block = self._format_score_feedback(result)
                improved_context = (
                    f"{feedback_block}\n\n{context_str}"
                    if context_str
                    else feedback_block
                )
                improve_parts: list[str] = []
                async for token in self._stream_synthesis(
                    query, improved_context, history, persona=persona
                ):
                    improve_parts.append(token)
                    yield {"type": "token", "content": token}
                new_answer = "".join(improve_parts)
                if new_answer:
                    answer = new_answer

        if answer:
            session_store.add_message(sid, Message(role="assistant", content=answer))
            await self._maybe_compress_memory(session_store, sid, http_client, aux_timeout)

        sources_payload = [
            {
                "content": s.get("content", ""),
                "doc_id": s.get("doc_id", ""),
                "score": s.get("score", 0.0),
            }
            for s in all_sources
        ]

        # 다음 턴을 위해 현재 참조 문서를 세션에 저장.
        if self._config.rag.agent.session_source_reuse and sources_payload:
            limit = self._config.rag.agent.session_source_reuse_limit
            set_last = getattr(session_store, "set_last_sources", None)
            if callable(set_last):
                set_last(sid, sources_payload[:limit])

        yield {"type": "sources", "sources": sources_payload}
        yield {"type": "done", "session_id": sid}

    @staticmethod
    def _format_score_feedback(result) -> str:
        """Phase 13 — scorer 결과를 재합성 프롬프트에 주입할 텍스트로."""
        lines: list[str] = ["[이전 답변 개선 지침]"]
        if result.feedback:
            lines.append(f"- 총평: {result.feedback}")
        for i, improvement in enumerate(result.improvements, start=1):
            lines.append(f"- 개선 {i}: {improvement}")
        return "\n".join(lines)

    async def _maybe_compress_memory(
        self,
        session_store: Any,
        sid: str,
        http_client: Any,
        aux_timeout: float,
    ) -> None:
        """Phase 12 — 세션 이력이 임계값을 넘으면 오래된 턴을 요약으로 압축 — never raises."""
        cfg = self._config.rag.agent
        if not getattr(cfg, "memory_compression_enabled", False):
            return
        compress_after = getattr(cfg, "compress_after_turns", 10)
        target_chars = getattr(cfg, "compress_target_chars", 500)

        # 현재 메시지 수 조회.
        try:
            _, msgs = session_store.get_or_create(sid)
        except Exception:
            return
        # compress_after_turns는 user+assistant 쌍 기준 → 메시지 수 2 * turns.
        threshold = compress_after * 2
        if len(msgs) <= threshold:
            return

        # 최신 (compress_after) 턴만 남기고 나머지를 요약.
        keep_recent = compress_after  # 메시지 개수
        to_summarize = msgs[:-keep_recent] if keep_recent > 0 else list(msgs)
        if not to_summarize:
            return

        from .memory import ConversationCompressor

        compressor = ConversationCompressor(
            http_client=http_client,
            ollama_model=self._model_for("verifier"),  # 가벼운 모델 활용
            api_base=self._api_base,
            request_timeout=aux_timeout,
            target_chars=target_chars,
        )
        try:
            summary = await compressor.summarize(to_summarize)
        except Exception as exc:
            logger.warning("Memory compression 실패: %s — 건너뜀", exc)
            return
        if not summary:
            return

        compress_fn = getattr(session_store, "compress_old_turns", None)
        if not callable(compress_fn):
            return
        removed = compress_fn(sid, keep_recent, summary)
        if removed:
            logger.info("세션 %s 압축: 메시지 %d개 → 요약 1개", sid, removed)

    def _format_active_skills(self, query: str) -> str:
        """질의에 매칭되는 skill들의 prompt_addon을 조립합니다. 없으면 빈 문자열."""
        if len(self._skill_registry) == 0:
            return ""
        matches = self._skill_registry.select_for_query(query, limit=3)
        if not matches:
            return ""
        return self._skill_registry.format_addons(matches)

    def _format_prior_context(self, session_store: Any, sid: str) -> str:
        """이전 턴의 참조 문서를 synthesis 프롬프트 컨텍스트로 포맷합니다."""
        if not self._config.rag.agent.session_source_reuse:
            return ""
        get_last = getattr(session_store, "get_last_sources", None)
        if not callable(get_last):
            return ""
        prior = get_last(sid)
        if not prior:
            return ""
        limit = self._config.rag.agent.session_source_reuse_limit
        lines = ["[이전 대화 참조 문서]"]
        for i, src in enumerate(prior[:limit], start=1):
            doc_id = src.get("doc_id", "?")
            content = src.get("content", "")[:300]
            lines.append(f"[이전 문서 {i}] (ID: {doc_id})\n{content}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 내부: 답변 합성 — Ollama 스트리밍
    # ------------------------------------------------------------------

    async def _stream_synthesis(
        self,
        query: str,
        context: str,
        history: str,
        *,
        persona: Persona | None = None,
    ) -> AsyncGenerator[str, None]:
        """수집된 컨텍스트로 최종 답변을 LLM에 스트리밍 요청합니다."""
        from .prompts import ANSWER_SYNTHESIS_PROMPT

        http_client = self._app_state.http_client
        if not context.strip():
            context = "(수집된 문서 없음)"

        template = (
            persona.synthesis_prompt_template
            if persona is not None and persona.synthesis_prompt_template
            else ANSWER_SYNTHESIS_PROMPT
        )
        prompt = template.format(
            history=f"{history}\n" if history else "",
            context=context[:_SYNTHESIS_CONTEXT_CHAR_LIMIT],
            query=query,
        )

        payload = {
            "model": self._model_for("synthesis"),
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": -1,
            "options": {"num_predict": self._rag_max_tokens},
        }
        try:
            async with http_client.stream(
                "POST",
                f"{self._api_base}/api/generate",
                json=payload,
                timeout=self._config.rag.request_timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.error("답변 합성 실패: %s", exc)
            yield "[오류] 답변 생성 중 문제가 발생했습니다."


__all__ = ["AgentOrchestrator", "SimpleStreamFn"]
