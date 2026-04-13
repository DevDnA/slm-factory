"""ReAct 에이전트 루프 — 질문 분석, 도구 사용, 답변 생성을 오케스트레이트합니다.

``AgentLoop``은 LLM과 도구를 반복적으로 호출하여 복합 질문에
다단계 검색·추론으로 답변합니다.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from ...utils import get_logger
from .parser import parse_react_output, _SPECIAL_TAG_RE
from .prompts import AGENT_SYSTEM_PROMPT, DECOMPOSE_PROMPT, FORCE_ANSWER_PROMPT
from .tools import ToolRegistry

logger = get_logger("rag.agent.loop")

# 스크래치패드 최대 길이 (프롬프트 폭발 방지)
_MAX_SCRATCHPAD_LEN = 6000
# Observation 최대 길이 (스크래치패드에 추가 시)
_MAX_OBSERVATION_LEN = 1500


# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """ReAct 루프 이벤트 — 스트리밍 시 클라이언트로 전송됩니다."""

    type: str  # "thought", "action", "observation", "token", "done", "error"
    content: str = ""
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """ReAct 루프 최종 결과."""

    answer: str
    sources: list[dict[str, Any]]
    iterations: int
    events: list[AgentEvent]


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """ReAct 에이전트 루프.

    Parameters
    ----------
    http_client:
        Ollama API 호출용 httpx.AsyncClient.
    tool_registry:
        에이전트가 사용할 도구 레지스트리.
    ollama_model:
        Ollama 모델명.
    api_base:
        Ollama API 베이스 URL.
    max_iterations:
        ReAct 루프 최대 반복 횟수.
    max_tokens:
        LLM 생성 최대 토큰 수. -1이면 무제한.
    request_timeout:
        LLM 요청 타임아웃(초).
    """

    def __init__(
        self,
        http_client: Any,
        tool_registry: ToolRegistry,
        ollama_model: str,
        api_base: str,
        max_iterations: int = 5,
        max_tokens: int = -1,
        request_timeout: float = 120.0,
    ) -> None:
        self._http_client = http_client
        self._tools = tool_registry
        self._model = ollama_model
        self._api_base = api_base
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(self, prompt: str) -> str:
        """Ollama /api/generate를 비스트리밍으로 호출하여 텍스트를 생성합니다."""
        import re

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": -1,
        }
        if self._max_tokens > 0:
            payload["options"] = {"num_predict": self._max_tokens}

        response = await self._http_client.post(
            f"{self._api_base}/api/generate",
            json=payload,
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    async def _generate_stream_tokens(
        self, prompt: str
    ) -> AsyncGenerator[str, None]:
        """Ollama /api/generate를 스트리밍으로 호출하여 토큰을 yield합니다."""
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": -1,
        }
        if self._max_tokens > 0:
            payload["options"] = {"num_predict": self._max_tokens}

        async with self._http_client.stream(
            "POST",
            f"{self._api_base}/api/generate",
            json=payload,
            timeout=self._request_timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token

    # ------------------------------------------------------------------
    # 프롬프트 조립
    # ------------------------------------------------------------------

    def _build_prompt(
        self, query: str, history: str, scratchpad: str
    ) -> str:
        """ReAct 시스템 프롬프트를 조립합니다."""
        conversation_history = f"{history}\n" if history else ""
        prompt = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self._tools.get_tool_descriptions(),
            max_iterations=self._max_iterations,
            conversation_history=conversation_history,
            query=query,
        )
        if scratchpad:
            # 스크래치패드 길이 제한 — 프롬프트 폭발 방지
            if len(scratchpad) > _MAX_SCRATCHPAD_LEN:
                scratchpad = "...(이전 단계 생략)\n" + scratchpad[-_MAX_SCRATCHPAD_LEN:]
            prompt += f"\n{scratchpad}"
        return prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _decompose_query(self, query: str) -> list[str]:
        """복합 질문을 하위 질문으로 분해합니다. 단순 질문이면 그대로 반환."""
        try:
            prompt = DECOMPOSE_PROMPT.format(query=query)
            response = await self._generate(prompt)
            # JSON 추출
            brace_start = response.find("{")
            brace_end = response.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                data = json.loads(response[brace_start:brace_end + 1])
                subs = data.get("sub_questions", [])
                if isinstance(subs, list) and len(subs) >= 1:
                    return subs[:3]  # 최대 3개
        except Exception:
            pass
        return [query]

    async def run(self, query: str, history: str = "") -> AgentResult:
        """동기적 실행 — 최종 결과를 반환합니다."""
        events: list[AgentEvent] = []
        all_sources: list[dict[str, Any]] = []
        scratchpad = ""

        # 질문 분해: 복합 질문이면 서브 질문별로 사전 검색
        sub_questions = await self._decompose_query(query)
        if len(sub_questions) > 1:
            logger.info("질문 분해: %d개 하위 질문", len(sub_questions))
            for sq in sub_questions:
                tool_result = await self._tools.execute("search", {"query": sq})
                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)
                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: 하위 질문 '{sq}'에 대해 검색합니다.\n"
                scratchpad += f"Action: search\n"
                scratchpad += f'Action Input: {{"query": "{sq}"}}\n'
                scratchpad += f"Observation: {obs_text}\n"
                events.append(AgentEvent(type="thought", content=f"하위 질문: {sq}", iteration=0))
                events.append(AgentEvent(type="observation", content=obs_text, iteration=0))

        for iteration in range(1, self._max_iterations + 1):
            prompt = self._build_prompt(query, history, scratchpad)

            try:
                llm_output = await self._generate(prompt)
            except Exception as e:
                logger.error("LLM 호출 실패 (iteration %d): %s", iteration, e)
                events.append(AgentEvent(
                    type="error", content="LLM 호출 중 문제가 발생했습니다.", iteration=iteration
                ))
                break

            step = parse_react_output(llm_output)

            # Thought 이벤트
            if step.thought:
                events.append(AgentEvent(
                    type="thought", content=step.thought, iteration=iteration
                ))

            # Final Answer — 루프 종료
            if step.final_answer is not None:
                events.append(AgentEvent(
                    type="done", content=step.final_answer, iteration=iteration,
                    metadata={"sources": all_sources},
                ))
                return AgentResult(
                    answer=step.final_answer,
                    sources=all_sources,
                    iterations=iteration,
                    events=events,
                )

            # Action — 도구 실행
            if step.action:
                events.append(AgentEvent(
                    type="action", content=step.action, iteration=iteration,
                    metadata={"input": step.action_input or {}},
                ))

                tool_result = await self._tools.execute(
                    step.action, step.action_input or {}
                )

                # 구조화된 소스 수집 (regex 파싱 불필요)
                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)

                events.append(AgentEvent(
                    type="observation", content=tool_result.text, iteration=iteration
                ))

                # 스크래치패드에 추가 (observation 길이 제한)
                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: {step.thought}\n"
                scratchpad += f"Action: {step.action}\n"
                scratchpad += f"Action Input: {json.dumps(step.action_input or {}, ensure_ascii=False)}\n"
                scratchpad += f"Observation: {obs_text}\n"
            else:
                # Action도 Final Answer도 없는 경우 — 답변으로 처리
                answer = step.thought or llm_output.strip()
                events.append(AgentEvent(
                    type="done", content=answer, iteration=iteration,
                    metadata={"sources": all_sources},
                ))
                return AgentResult(
                    answer=answer,
                    sources=all_sources,
                    iterations=iteration,
                    events=events,
                )

        # max_iterations 도달 — 강제 답변 생성
        return await self._force_answer(query, scratchpad, all_sources, events)

    async def run_stream(
        self, query: str, history: str = ""
    ) -> AsyncGenerator[AgentEvent, None]:
        """스트리밍 실행 — Final Answer 감지 후 실시간 토큰 전달."""
        import re

        all_sources: list[dict[str, Any]] = []
        scratchpad = ""

        _FINAL_ANSWER_MARKER = re.compile(
            r"(?:Final Answer|최종 답변)\s*:\s*", re.IGNORECASE
        )
        _REASONING_STOP = re.compile(
            r"\n\s*(?:Thought|생각|Action|행동|Observation|관찰)\s*:", re.IGNORECASE
        )

        # 질문 분해: 복합 질문이면 서브 질문별로 사전 검색
        sub_questions = await self._decompose_query(query)
        if len(sub_questions) > 1:
            logger.info("질문 분해: %d개 하위 질문", len(sub_questions))
            for sq in sub_questions:
                yield AgentEvent(type="thought", content=f"하위 질문: {sq}", iteration=0)
                tool_result = await self._tools.execute("search", {"query": sq})
                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)
                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: 하위 질문 '{sq}'에 대해 검색합니다.\n"
                scratchpad += f"Action: search\n"
                scratchpad += f'Action Input: {{"query": "{sq}"}}\n'
                scratchpad += f"Observation: {obs_text}\n"
                yield AgentEvent(type="observation", content=obs_text[:300], iteration=0)

        for iteration in range(1, self._max_iterations + 1):
            prompt = self._build_prompt(query, history, scratchpad)

            # 스트리밍으로 토큰을 받으면서 Final Answer를 감지
            buffer = ""
            found_final = False
            try:
                async for token in self._generate_stream_tokens(prompt):
                    buffer += token

                    if not found_final:
                        # Final Answer: 패턴 감지
                        m = _FINAL_ANSWER_MARKER.search(buffer)
                        if m:
                            found_final = True
                            # 패턴 앞부분에서 Thought 추출
                            pre_text = buffer[:m.start()]
                            step = parse_react_output(pre_text)
                            if step.thought:
                                yield AgentEvent(
                                    type="thought", content=step.thought,
                                    iteration=iteration,
                                )
                            # 패턴 뒤의 텍스트를 즉시 전달
                            after = _SPECIAL_TAG_RE.sub("", buffer[m.end():])
                            if after:
                                yield AgentEvent(
                                    type="token", content=after,
                                    iteration=iteration,
                                )
                    else:
                        # Final Answer 이후 — 추론 키워드 나오면 중단
                        if _REASONING_STOP.search(buffer[buffer.rfind("Final Answer"):] if "Final Answer" in buffer else buffer[-200:]):
                            break
                        # 특수 태그 필터링
                        clean_token = _SPECIAL_TAG_RE.sub("", token)
                        if clean_token:
                            yield AgentEvent(
                                type="token", content=clean_token,
                                iteration=iteration,
                            )

            except Exception as e:
                logger.error("LLM 호출 실패 (iteration %d): %s", iteration, e)
                yield AgentEvent(
                    type="error", content="LLM 호출 중 문제가 발생했습니다.",
                    iteration=iteration,
                )
                return

            # thinking 태그 제거
            buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL).strip()

            if found_final:
                # 스트리밍 완료
                yield AgentEvent(
                    type="done", content="", iteration=iteration,
                    metadata={"sources": all_sources},
                )
                return

            # Final Answer가 없었으면 — Thought/Action 파싱
            step = parse_react_output(buffer)

            if step.thought:
                yield AgentEvent(
                    type="thought", content=step.thought, iteration=iteration,
                )

            if step.action:
                yield AgentEvent(
                    type="action", content=step.action, iteration=iteration,
                    metadata={"input": step.action_input or {}},
                )

                tool_result = await self._tools.execute(
                    step.action, step.action_input or {}
                )

                for src in tool_result.sources:
                    if not any(s.get("doc_id") == src.get("doc_id") for s in all_sources):
                        all_sources.append(src)

                yield AgentEvent(
                    type="observation", content=tool_result.text,
                    iteration=iteration,
                )

                obs_text = tool_result.text
                if len(obs_text) > _MAX_OBSERVATION_LEN:
                    obs_text = obs_text[:_MAX_OBSERVATION_LEN] + "\n...(결과 생략)"
                scratchpad += f"\nThought: {step.thought}\n"
                scratchpad += f"Action: {step.action}\n"
                scratchpad += f"Action Input: {json.dumps(step.action_input or {}, ensure_ascii=False)}\n"
                scratchpad += f"Observation: {obs_text}\n"
            else:
                # Action도 Final Answer도 없음 — fallback 답변
                answer = step.thought or buffer
                if answer:
                    yield AgentEvent(
                        type="token", content=answer, iteration=iteration,
                    )
                yield AgentEvent(
                    type="done", content="", iteration=iteration,
                    metadata={"sources": all_sources},
                )
                return

        # max_iterations 도달 — 강제 답변도 실시간 스트리밍
        logger.warning("max_iterations(%d) 도달 — 강제 답변 생성", self._max_iterations)
        force_prompt = FORCE_ANSWER_PROMPT.format(
            gathered_context=scratchpad[-3000:] if scratchpad else "(수집된 정보 없음)",
            query=query,
        )
        try:
            async for token in self._generate_stream_tokens(force_prompt):
                yield AgentEvent(
                    type="token", content=token,
                    iteration=self._max_iterations,
                )
        except Exception as e:
            logger.error("강제 답변 생성 실패: %s", e)
            yield AgentEvent(
                type="token",
                content="답변 생성 중 문제가 발생했습니다. 다시 시도해 주세요.",
                iteration=self._max_iterations,
            )
        yield AgentEvent(
            type="done", content="", iteration=self._max_iterations,
            metadata={"sources": all_sources},
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    async def _force_answer(
        self,
        query: str,
        scratchpad: str,
        all_sources: list[dict],
        events: list[AgentEvent],
    ) -> AgentResult:
        """max_iterations 도달 시 수집된 컨텍스트로 강제 답변을 생성합니다."""
        logger.warning("max_iterations(%d) 도달 — 강제 답변 생성", self._max_iterations)

        prompt = FORCE_ANSWER_PROMPT.format(
            gathered_context=scratchpad[-3000:] if scratchpad else "(수집된 정보 없음)",
            query=query,
        )

        try:
            answer = await self._generate(prompt)
        except Exception as e:
            logger.error("강제 답변 생성 실패: %s", e)
            answer = "답변 생성 중 문제가 발생했습니다. 다시 시도해 주세요."

        events.append(AgentEvent(
            type="done", content=answer, iteration=self._max_iterations,
            metadata={"sources": all_sources, "forced": True},
        ))
        return AgentResult(
            answer=answer,
            sources=all_sources,
            iterations=self._max_iterations,
            events=events,
        )

def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """텍스트를 지정된 크기의 청크로 분할합니다."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
