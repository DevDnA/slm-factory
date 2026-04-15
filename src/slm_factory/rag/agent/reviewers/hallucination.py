"""HallucinationChecker — sources 외 주장(환각) 검출."""

from __future__ import annotations

from ..prompts import HALLUCINATION_CHECK_PROMPT
from .base import Reviewer


class HallucinationChecker(Reviewer):
    name = "hallucination"
    prompt_template = HALLUCINATION_CHECK_PROMPT


__all__ = ["HallucinationChecker"]
