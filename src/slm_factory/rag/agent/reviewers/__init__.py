"""Review-work 시스템 — 답변 사후 병렬 검증.

oh-my-openagent의 review-work 패턴을 RAG Q&A 컨텍스트에 이식:
3개의 특화 reviewer가 독립적으로 답변을 검증하고 aggregator가 종합 판정.

Reviewer
--------
- ``GroundingChecker``: 답변의 주장이 sources에 실제로 근거하는가?
- ``CompletenessChecker``: 질문의 모든 부분에 답했는가?
- ``HallucinationChecker``: sources에 없는 주장이 있는가?

모든 reviewer는 ``never-raise`` 계약. LLM 실패 시 ``passed=True``로 fallback.
"""

from __future__ import annotations

from .aggregator import AggregatedVerdict, run_reviewers
from .base import ReviewVerdict, Reviewer
from .completeness import CompletenessChecker
from .grounding import GroundingChecker
from .hallucination import HallucinationChecker

__all__ = [
    "AggregatedVerdict",
    "CompletenessChecker",
    "GroundingChecker",
    "HallucinationChecker",
    "ReviewVerdict",
    "Reviewer",
    "run_reviewers",
]
