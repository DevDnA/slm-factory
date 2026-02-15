"""QA 쌍을 멀티턴 대화로 확장하는 대화 생성기."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from ..config import DialogueConfig, TeacherConfig
    from .base import BaseTeacher

from ..models import DialogueTurn, MultiTurnDialogue, QAPair
from ..utils import get_logger

logger = get_logger("dialogue_generator")


class DialogueGenerator:
    """QA 쌍을 멀티턴 대화로 확장합니다."""

    def __init__(
        self,
        teacher: BaseTeacher,
        config: DialogueConfig,
        teacher_config: TeacherConfig,
    ) -> None:
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config

    def _build_dialogue_prompt(self, pair: QAPair) -> str:
        """대화 생성을 위한 프롬프트를 구성합니다."""
        return (
            "다음 질문-답변 쌍을 기반으로 자연스러운 멀티턴 대화를 생성해주세요.\n\n"
            f"## 원본 QA\n"
            f"질문: {pair.question}\n"
            f"답변: {pair.answer}\n\n"
            f"## 규칙\n"
            f"- 대화는 {self.config.min_turns}~{self.config.max_turns}턴으로 구성합니다\n"
            "- 각 턴은 user와 assistant 역할이 번갈아 나옵니다\n"
            "- 첫 번째 턴은 반드시 user 역할이어야 합니다\n"
            "- 원본 QA의 내용을 자연스럽게 포함해야 합니다\n"
            "- user는 후속 질문이나 관련 질문을 할 수 있습니다\n"
            "- assistant는 정확하고 유용한 답변을 제공합니다\n\n"
            '반드시 아래 JSON 형식으로만 응답하세요:\n'
            '{"turns": [{"role": "user", "content": "..."}, '
            '{"role": "assistant", "content": "..."}, ...]}'
        )

    def _parse_dialogue(
        self, text: str, pair: QAPair
    ) -> MultiTurnDialogue | None:
        """LLM 응답에서 대화를 파싱합니다."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            raw_turns = data.get("turns", [])
            if not isinstance(raw_turns, list) or len(raw_turns) < 2:
                logger.warning("대화 턴 수 부족: %d", len(raw_turns) if isinstance(raw_turns, list) else 0)
                return None

            turns: list[DialogueTurn] = []
            for t in raw_turns:
                role = str(t.get("role", ""))
                content = str(t.get("content", ""))
                if role in ("user", "assistant") and content:
                    turns.append(DialogueTurn(role=role, content=content))

            if len(turns) < 2:
                logger.warning("유효한 턴 수 부족: %d", len(turns))
                return None

            # min_turns/max_turns 적용
            if len(turns) < self.config.min_turns:
                logger.warning(
                    "턴 수(%d)가 min_turns(%d)보다 적음",
                    len(turns),
                    self.config.min_turns,
                )
                return None

            if len(turns) > self.config.max_turns:
                turns = turns[: self.config.max_turns]

            return MultiTurnDialogue(
                turns=turns,
                source_doc=pair.source_doc,
                category=pair.category,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("대화 파싱 실패: %s", e)
            return None

    async def generate_dialogue(self, pair: QAPair) -> MultiTurnDialogue | None:
        """단일 QA 쌍에서 멀티턴 대화를 생성합니다."""
        prompt = self._build_dialogue_prompt(pair)

        kwargs: dict[str, Any] = {}
        if self.teacher_config.backend == "ollama":
            kwargs["format"] = "json"

        response = await self.teacher.agenerate(prompt, **kwargs)
        return self._parse_dialogue(response, pair)

    async def generate_all(
        self, pairs: list[QAPair]
    ) -> list[MultiTurnDialogue]:
        """전체 QA 쌍에 대해 멀티턴 대화를 생성합니다."""
        semaphore = asyncio.Semaphore(self.teacher_config.max_concurrency)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        dialogues: list[MultiTurnDialogue] = []
        failed = 0

        with progress:
            task_id = progress.add_task("멀티턴 대화 생성 중...", total=len(pairs))

            async def _bounded_generate(
                pair: QAPair,
            ) -> MultiTurnDialogue | None:
                async with semaphore:
                    result = await self.generate_dialogue(pair)
                    progress.advance(task_id)
                    return result

            tasks = [_bounded_generate(pair) for pair in pairs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.error("대화 생성 실패: %s", result)
                failed += 1
                continue
            if result is None:
                failed += 1
                continue
            dialogues.append(result)

        logger.info(
            "멀티턴 대화 생성 완료: %d/%d 성공 (%d 실패)",
            len(dialogues),
            len(pairs),
            failed,
        )
        return dialogues

    def save_dialogues(
        self, dialogues: list[MultiTurnDialogue], path: Path
    ) -> None:
        """대화 데이터를 JSON 파일로 저장합니다."""
        data = [asdict(d) for d in dialogues]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("%d개 대화를 %s에 저장함", len(dialogues), path)
