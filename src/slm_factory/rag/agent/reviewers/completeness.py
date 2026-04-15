"""CompletenessChecker — 질문의 모든 부분에 답했는지 검증."""

from __future__ import annotations

from ..prompts import COMPLETENESS_CHECK_PROMPT
from .base import Reviewer


class CompletenessChecker(Reviewer):
    name = "completeness"
    prompt_template = COMPLETENESS_CHECK_PROMPT


__all__ = ["CompletenessChecker"]
