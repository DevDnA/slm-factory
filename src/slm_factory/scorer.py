"""교사 LLM을 사용한 QA 쌍 품질 점수 평가."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ScoringConfig, TeacherConfig
    from .teacher.base import BaseTeacher

from .models import QAPair
from .utils import get_logger

logger = get_logger("scorer")


class QualityScorer:
    """교사 LLM을 사용하여 QA 쌍의 품질을 1~5점으로 평가합니다."""

    def __init__(self, teacher: BaseTeacher, config: ScoringConfig, teacher_config: TeacherConfig):
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config

    def _build_scoring_prompt(self, pair: QAPair) -> str:
        """점수 평가를 위한 프롬프트를 구성합니다."""
        return (
            "다음 질문-답변 쌍의 품질을 1~5점으로 평가해주세요.\n\n"
            "## 평가 기준\n"
            "- 1점: 답변이 완전히 잘못되었거나 무관합니다\n"
            "- 2점: 답변이 부분적으로 맞지만 심각한 오류가 있습니다\n"
            "- 3점: 답변이 대체로 맞지만 불완전하거나 부정확한 부분이 있습니다\n"
            "- 4점: 답변이 정확하고 충분하지만 약간의 개선 여지가 있습니다\n"
            "- 5점: 답변이 정확하고 완전하며 잘 구조화되어 있습니다\n\n"
            "## 평가 대상\n"
            f"질문: {pair.question}\n"
            f"답변: {pair.answer}\n\n"
            "## 참고 예시\n"
            "예시 1 - 5점: 질문에 정확히 답하고, 구체적 수치/날짜를 포함하며, 논리적으로 구조화됨\n"
            "예시 2 - 2점: 질문과 관련은 있지만 핵심을 벗어나거나 회피적 답변\n"
            "예시 3 - 1점: 질문과 무관한 답변이거나 명백한 오류 포함\n\n"
            '반드시 아래 JSON 형식으로만 응답하세요:\n'
            '{"score": <1-5 정수>, "reason": "<평가 근거 한 문장>"}'
        )

    def _parse_score(self, text: str) -> tuple[int, str] | None:
        """LLM 응답에서 점수와 이유를 추출합니다."""
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            data = json.loads(text)
            score = int(data.get("score", 0))
            reason = str(data.get("reason", ""))
            if 1 <= score <= 5:
                return score, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        match = re.search(r'[1-5]', text)
        if match:
            return int(match.group()), "점수만 추출됨"

        logger.warning("점수 파싱 실패: %s", text[:100])
        return None

    async def score_one(self, pair: QAPair) -> tuple[QAPair, int, str]:
        """단일 QA 쌍을 점수 평가합니다."""
        prompt = self._build_scoring_prompt(pair)

        kwargs: dict[str, Any] = {}
        if self.teacher_config.backend == "ollama":
            kwargs["format"] = "json"

        response = await self.teacher.agenerate(prompt, **kwargs)
        result = self._parse_score(response)

        if result is None:
            return pair, 3, "점수 파싱 실패 — 기본값 3 적용"

        score, reason = result
        return pair, score, reason

    async def score_all(
        self,
        pairs: list[QAPair],
    ) -> tuple[list[QAPair], list[tuple[QAPair, int, str]]]:
        """전체 QA 쌍을 점수 평가하고 threshold 기준으로 필터링합니다."""
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def _bounded_score(pair: QAPair) -> tuple[QAPair, int, str]:
            async with semaphore:
                return await self.score_one(pair)

        tasks = [_bounded_score(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        accepted: list[QAPair] = []
        filtered: list[tuple[QAPair, int, str]] = []

        for result in results:
            if isinstance(result, Exception):
                logger.error("점수 평가 실패: %s", result)
                continue
            pair, score, reason = result
            if score >= self.config.threshold:
                accepted.append(pair)
            else:
                filtered.append((pair, score, reason))
                logger.debug(
                    "QA 쌍 제거 (점수 %d < %.0f): Q=%s... 이유: %s",
                    score, self.config.threshold, pair.question[:40], reason,
                )

        logger.info(
            "품질 점수 평가 완료: %d/%d 통과 (threshold=%.1f), %d 제거",
            len(accepted), len(pairs), self.config.threshold, len(filtered),
        )
        return accepted, filtered
