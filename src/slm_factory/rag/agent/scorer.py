"""AnswerScorer — 답변 품질을 1~10 정량 점수로 평가.

Reflector (yes/no 판정)와 달리 Scorer는 수치적 점수를 반환하여 iterative
개선 루프(Ralph pattern)를 구동할 수 있습니다. 낮은 점수일 때 구체적
피드백을 추출하여 다음 생성 시 프롬프트에 주입.

설계 원칙
---------
- **Never-raise**: LLM·파싱 실패 시 중립 점수(7)로 반환.
- **클리핑**: 점수는 1~10 범위로 정규화.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ...utils import get_logger
from .prompts import ANSWER_SCORER_PROMPT

logger = get_logger("rag.agent.scorer")

_ANSWER_CHAR_LIMIT = 2000
_SOURCES_PREVIEW_LIMIT = 1000
# LLM이 점수를 파싱 실패했을 때 사용할 중립값 — 재시도 유발 최소화.
_NEUTRAL_SCORE = 7.0


@dataclass
class ScoreResult:
    """AnswerScorer 판정 결과."""

    score: float  # 1.0 ~ 10.0
    feedback: str = ""
    improvements: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.improvements is None:
            self.improvements = []

    def below(self, threshold: float) -> bool:
        """점수가 threshold 미만인지."""
        return self.score < threshold


class AnswerScorer:
    """답변에 1~10 정량 점수 + 개선 피드백을 부여합니다.

    Parameters
    ----------
    http_client:
        Ollama ``/api/generate`` 호출용.
    ollama_model:
        평가 모델.
    api_base:
        Ollama API 베이스 URL.
    request_timeout:
        요청 타임아웃(초).
    max_tokens:
        생성 최대 토큰.
    """

    def __init__(
        self,
        http_client: Any,
        ollama_model: str,
        api_base: str,
        request_timeout: float = 20.0,
        max_tokens: int = 300,
    ) -> None:
        self._http_client = http_client
        self._model = ollama_model
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens

    async def score(
        self,
        query: str,
        answer: str,
        sources: list[dict] | None = None,
    ) -> ScoreResult:
        """답변을 1~10 점수로 평가 — never raises."""
        if not answer.strip():
            return ScoreResult(
                score=1.0,
                feedback="빈 답변",
                improvements=["답변이 생성되지 않음 — 재생성 필요"],
            )

        try:
            raw = await self._generate(query, answer, sources or [])
        except Exception as exc:
            logger.warning("Scorer LLM 호출 실패: %s — 중립 점수", exc)
            return ScoreResult(
                score=_NEUTRAL_SCORE, feedback="scorer unavailable"
            )

        parsed = self._parse(raw)
        if parsed is None:
            logger.debug("Scorer JSON 파싱 실패 — 중립 점수")
            return ScoreResult(score=_NEUTRAL_SCORE, feedback="parse failure")

        return self._to_result(parsed)

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------

    async def _generate(
        self, query: str, answer: str, sources: list[dict]
    ) -> str:
        preview = self._format_sources(sources)
        prompt = ANSWER_SCORER_PROMPT.format(
            query=query,
            answer=answer[:_ANSWER_CHAR_LIMIT],
            sources_preview=preview,
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
        for i, s in enumerate(sources, start=1):
            if not isinstance(s, dict):
                continue
            did = s.get("doc_id", "?")
            c = str(s.get("content", ""))[:150]
            line = f"[문서 {i}] (ID: {did}) {c}"
            if total + len(line) > _SOURCES_PREVIEW_LIMIT:
                lines.append("...(생략)")
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

    @staticmethod
    def _to_result(data: dict) -> ScoreResult:
        raw_score = data.get("score", _NEUTRAL_SCORE)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = _NEUTRAL_SCORE
        score = max(1.0, min(10.0, score))

        feedback = str(data.get("feedback", ""))[:400]
        raw_imp = data.get("improvements") or []
        if not isinstance(raw_imp, list):
            raw_imp = []
        improvements = [str(i)[:200] for i in raw_imp if i]

        return ScoreResult(
            score=score,
            feedback=feedback,
            improvements=improvements,
        )


__all__ = ["AnswerScorer", "ScoreResult"]
