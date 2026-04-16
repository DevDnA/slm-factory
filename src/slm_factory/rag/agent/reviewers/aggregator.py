"""Reviewer 종합 실행기 — asyncio.gather로 3 reviewer 병렬 실행."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ....utils import get_logger
from .base import ReviewVerdict, Reviewer
from .completeness import CompletenessChecker
from .grounding import GroundingChecker
from .hallucination import HallucinationChecker

logger = get_logger("rag.agent.reviewers.aggregator")


@dataclass
class AggregatedVerdict:
    """3 reviewer의 종합 판정."""

    overall_passed: bool
    verdicts: list[ReviewVerdict] = field(default_factory=list)
    missing_info_query: str | None = None

    @property
    def needs_retry(self) -> bool:
        return not self.overall_passed and bool(self.missing_info_query)

    @property
    def failed_reviewers(self) -> list[str]:
        return [v.reviewer for v in self.verdicts if not v.passed]


async def run_reviewers(
    query: str,
    answer: str,
    sources: list[dict],
    *,
    http_client: Any,
    ollama_model: str,
    api_base: str,
    request_timeout: float = 20.0,
    reviewers: list[Reviewer] | None = None,
    keep_alive: str = "5m",
) -> AggregatedVerdict:
    """3 reviewer를 병렬로 실행하고 종합 판정을 반환합니다 — never raises.

    Parameters
    ----------
    reviewers:
        사용할 Reviewer 인스턴스 목록. ``None``이면 기본 3개(grounding,
        completeness, hallucination).
    keep_alive:
        Ollama ``keep_alive`` 파라미터. reviewers가 ``None``일 때만 사용됨
        (사용자 지정 reviewers는 자체 keep_alive를 가짐).
    """
    if reviewers is None:
        reviewers = [
            GroundingChecker(
                http_client=http_client,
                ollama_model=ollama_model,
                api_base=api_base,
                request_timeout=request_timeout,
                keep_alive=keep_alive,
            ),
            CompletenessChecker(
                http_client=http_client,
                ollama_model=ollama_model,
                api_base=api_base,
                request_timeout=request_timeout,
                keep_alive=keep_alive,
            ),
            HallucinationChecker(
                http_client=http_client,
                ollama_model=ollama_model,
                api_base=api_base,
                request_timeout=request_timeout,
                keep_alive=keep_alive,
            ),
        ]

    try:
        results = await asyncio.gather(
            *[r.check(query, answer, sources) for r in reviewers],
            return_exceptions=True,
        )
    except Exception as exc:  # pragma: no cover — reviewer는 자체 never-raise
        logger.warning("Reviewer 병렬 실행 오류: %s — 전체 pass 처리", exc)
        return AggregatedVerdict(overall_passed=True)

    verdicts: list[ReviewVerdict] = []
    for r, result in zip(reviewers, results):
        if isinstance(result, Exception):
            logger.warning("Reviewer '%s' 예외: %s — pass 처리", r.name, result)
            verdicts.append(
                ReviewVerdict(reviewer=r.name, passed=True, reason="reviewer exception — assumed pass")
            )
        else:
            verdicts.append(result)

    overall_passed = all(v.passed for v in verdicts)

    # 첫 번째 retry hint 우선 — missing_info가 있는 첫 실패 verdict 선택.
    missing_query: str | None = None
    for v in verdicts:
        if v.has_retry_hint:
            missing_query = v.missing_info
            break

    return AggregatedVerdict(
        overall_passed=overall_passed,
        verdicts=verdicts,
        missing_info_query=missing_query,
    )


__all__ = ["AggregatedVerdict", "run_reviewers"]
