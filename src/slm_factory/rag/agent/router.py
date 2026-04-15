"""질의 복잡도 기반 라우팅 결정기.

``QueryRouter``는 사용자 질의를 검사하여 단순 RAG(``simple``) 또는
Agent RAG(``agent``) 경로 중 하나를 선택합니다. 현재 휴리스틱은
server.py에 인라인되어 있던 ``_is_complex_query`` 규칙을 그대로
옮겨와 동작이 동일함을 보장합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .intent_classifier import IntentCategory, IntentClassifier

RouteMode = Literal["simple", "agent"]


# 복합 질문 판단용 키워드 — 비교, 변경 이력, 인과, 조건 등.
_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "비교", "차이", "다른점", "대비", "vs", "versus",
    "변경", "바뀐", "달라진", "개정",
    "왜", "이유", "원인", "근거",
    "어떤 경우", "조건", "해당하는", "자격",
    "장단점", "장점과 단점", "pros", "cons",
    "관계", "연관", "영향", "따르면",
    "종합", "분석", "검토",
)

# "A와 B의 차이", "A와 B를 비교" 형태의 비교 절 패턴.
_COMPARISON_RE = re.compile(r".{2,}[와과].{2,}(의|을|를|에|는|가)")


@dataclass(frozen=True)
class RouteDecision:
    """라우팅 결정 결과.

    Attributes
    ----------
    mode:
        ``"simple"`` 또는 ``"agent"``.
    reason:
        라우팅 근거 — 로깅 및 디버깅용 사람이 읽을 수 있는 설명.
    complexity:
        감지된 복잡도 — 0.0 (단순) ~ 1.0 (복합).
    matched_keyword:
        복합으로 분류된 경우 첫 매칭 키워드. 단순일 땐 ``None``.
    intent:
        ``IntentClassifier``가 활성화된 경우 감지된 의도 카테고리.
        키워드 기반 라우팅만 사용한 경우 ``None``.
    """

    mode: RouteMode
    reason: str
    complexity: float
    matched_keyword: str | None = None
    intent: "IntentCategory | None" = None


class QueryRouter:
    """질의 복잡도 휴리스틱 + 선택적 LLM 의도 분류 기반 라우터.

    Parameters
    ----------
    agent_enabled:
        Agent 모드가 전반적으로 활성화되어 있는지. ``False``이면
        어떤 질의든 ``simple``로 라우팅됩니다.
    intent_classifier:
        선택적 LLM 기반 분류기. 제공되면 ``route_async()``가 먼저
        사용하고, 실패 시 키워드 휴리스틱으로 fallback.
    """

    def __init__(
        self,
        agent_enabled: bool = True,
        intent_classifier: "IntentClassifier | None" = None,
    ) -> None:
        self._agent_enabled = agent_enabled
        self._classifier = intent_classifier

    def route(self, query: str) -> RouteDecision:
        """주어진 질의의 라우팅을 결정합니다 (동기·키워드 전용)."""
        if not self._agent_enabled:
            return RouteDecision(
                mode="simple",
                reason="agent mode disabled",
                complexity=0.0,
            )

        lowered = query.lower()

        for keyword in _COMPLEX_KEYWORDS:
            if keyword in lowered:
                return RouteDecision(
                    mode="agent",
                    reason=f"복합 키워드 '{keyword}' 감지",
                    complexity=0.8,
                    matched_keyword=keyword,
                )

        if query.count("?") >= 2 or query.count("？") >= 2:
            return RouteDecision(
                mode="agent",
                reason="다중 물음표 — 복합 질문으로 판단",
                complexity=0.7,
            )

        if _COMPARISON_RE.search(query):
            return RouteDecision(
                mode="agent",
                reason="비교 절 패턴(~와/과 ~) 감지",
                complexity=0.75,
            )

        return RouteDecision(
            mode="simple",
            reason="복합 신호 없음 — 단순 RAG",
            complexity=0.1,
        )

    async def route_async(self, query: str) -> RouteDecision:
        """비동기 라우팅 — ``IntentClassifier``가 주입되어 있으면 사용.

        LLM 분류가 실패하거나 ``ambiguous`` 저신뢰로 반환되면 키워드 휴리스틱으로
        fallback합니다. 키워드가 ``agent``로 분류했고 LLM은 ``factual``이라면
        키워드 신호를 우선시합니다 (명시적 비교·분석 키워드가 LLM 모호성보다 강함).
        """
        if not self._agent_enabled or self._classifier is None:
            return self.route(query)

        keyword_decision = self.route(query)

        try:
            intent = await self._classifier.classify(query)
        except Exception:  # pragma: no cover — classifier는 자체 never-raise
            return keyword_decision

        # LLM이 명확한 복합 의도를 반환하면 agent로.
        llm_mode: RouteMode
        if intent.intent == "factual" and intent.confidence >= 0.7:
            llm_mode = "simple"
        else:
            llm_mode = "agent"

        # 키워드가 agent라고 판단한 경우 보수적으로 agent 유지
        # (명시적 비교·분석 키워드가 LLM보다 강한 신호).
        if keyword_decision.mode == "agent":
            return RouteDecision(
                mode="agent",
                reason=(
                    f"{keyword_decision.reason} + intent={intent.intent} "
                    f"conf={intent.confidence:.2f}"
                ),
                complexity=max(keyword_decision.complexity, intent.confidence),
                matched_keyword=keyword_decision.matched_keyword,
                intent=intent.intent,
            )

        return RouteDecision(
            mode=llm_mode,
            reason=f"intent={intent.intent} conf={intent.confidence:.2f} ({intent.reason})",
            complexity=intent.confidence if llm_mode == "agent" else 1.0 - intent.confidence,
            matched_keyword=None,
            intent=intent.intent,
        )


__all__ = ["QueryRouter", "RouteDecision", "RouteMode"]
