"""Reviewer 공통 기반 클래스 + ReviewVerdict."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ....utils import get_logger

logger = get_logger("rag.agent.reviewers.base")

# 프롬프트 폭발 방지.
_ANSWER_CHAR_LIMIT = 2000
_SOURCES_PREVIEW_LIMIT = 1500


@dataclass
class ReviewVerdict:
    """단일 reviewer의 판정 결과."""

    reviewer: str
    passed: bool
    reason: str = ""
    missing_info: str | None = None

    @property
    def has_retry_hint(self) -> bool:
        return not self.passed and bool(self.missing_info)


class Reviewer:
    """Reviewer 기반 클래스 — LLM JSON 호출 + 파싱 + fallback 공통 로직.

    서브클래스는 ``name``, ``prompt_template`` (class attr)을 정의하고
    ``_interpret(data: dict) -> ReviewVerdict``를 구현합니다.
    """

    name: str = "base-reviewer"
    prompt_template: str = ""

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 20.0,
        max_tokens: int = 200,
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        query: str,
        answer: str,
        sources: list[dict],
    ) -> ReviewVerdict:
        """답변을 검증하고 ``ReviewVerdict``를 반환합니다 — never raises."""
        if not answer.strip():
            return ReviewVerdict(
                reviewer=self.name,
                passed=False,
                reason="empty-answer",
                missing_info=query,
            )

        try:
            raw = await self._generate(query, answer, sources)
        except Exception as exc:
            logger.warning("%s LLM 호출 실패: %s — pass 처리", self.name, exc)
            return ReviewVerdict(
                reviewer=self.name,
                passed=True,
                reason="reviewer unavailable — assumed pass",
            )

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("%s JSON 파싱 실패 — pass 처리", self.name)
            return ReviewVerdict(
                reviewer=self.name,
                passed=True,
                reason="reviewer parse failure — assumed pass",
            )

        return self._interpret(parsed)

    # ------------------------------------------------------------------
    # 서브클래스 hook
    # ------------------------------------------------------------------

    def _interpret(self, data: dict) -> ReviewVerdict:
        """파싱된 dict를 ``ReviewVerdict``로 변환. 서브클래스가 override 가능."""
        raw_passed = data.get("passed")
        if raw_passed is None:
            raw_passed = data.get("ok")
        if isinstance(raw_passed, str):
            passed = raw_passed.strip().lower() in ("true", "yes", "예", "ok", "pass")
        else:
            passed = bool(raw_passed) if raw_passed is not None else True

        reason = str(data.get("reason", ""))[:300]
        raw_missing = data.get("missing_info") or data.get("next_query")
        missing = str(raw_missing).strip() if raw_missing else None
        if missing == "":
            missing = None

        return ReviewVerdict(
            reviewer=self.name,
            passed=passed,
            reason=reason,
            missing_info=missing,
        )

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------

    async def _generate(
        self, query: str, answer: str, sources: list[dict]
    ) -> str:
        sources_preview = self._format_sources(sources)
        prompt = self.prompt_template.format(
            query=query,
            answer=answer[:_ANSWER_CHAR_LIMIT],
            sources_preview=sources_preview,
        )
        response = await self._http_client.post(
            f"{self._api_base}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": "json",
                "keep_alive": -1,
                "options": {"num_predict": self._max_tokens},
            },
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "") or data.get("thinking", "")

    @staticmethod
    def _format_sources(sources: list[dict]) -> str:
        if not sources:
            return "(참고 문서 없음)"
        lines: list[str] = []
        total = 0
        for i, src in enumerate(sources, start=1):
            if not isinstance(src, dict):
                continue
            doc_id = src.get("doc_id", "?")
            content = str(src.get("content", ""))[:200]
            line = f"[문서 {i}] (ID: {doc_id}) {content}"
            if total + len(line) > _SOURCES_PREVIEW_LIMIT:
                lines.append("...(일부 생략)")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> dict | None:
        if not raw:
            return None
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start == -1 or brace_end <= brace_start:
            return None
        try:
            parsed = json.loads(raw[brace_start : brace_end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


__all__ = ["Reviewer", "ReviewVerdict"]
