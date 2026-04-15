"""GroundingChecker — 답변 주장의 sources 근거성 검증."""

from __future__ import annotations

from ..prompts import GROUNDING_CHECK_PROMPT
from .base import Reviewer


class GroundingChecker(Reviewer):
    name = "grounding"
    prompt_template = GROUNDING_CHECK_PROMPT


__all__ = ["GroundingChecker"]
