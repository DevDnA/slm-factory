"""교사 LLM을 사용한 온톨로지 엔티티·관계 추출기입니다.

문서에서 엔티티(개체)와 관계(트리플)를 자동 추출합니다.
QualityScorer 패턴을 따릅니다: 생성자 → 프롬프트 빌드 → JSON 파싱 →
비동기 단건/일괄 처리.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from ..config import OntologyConfig, TeacherConfig
    from ..models import ParsedDocument
    from ..teacher.base import BaseTeacher

from ..teacher.qa_generator import chunk_document
from ..utils import get_logger, run_bounded
from .models import Entity, KnowledgeGraph, Relation

logger = get_logger("ontology.extractor")


class OntologyExtractor:
    """교사 LLM을 사용하여 문서에서 엔티티와 관계를 추출합니다."""

    def __init__(
        self,
        teacher: BaseTeacher,
        config: OntologyConfig,
        teacher_config: TeacherConfig,
    ) -> None:
        self.teacher = teacher
        self.config = config
        self.teacher_config = teacher_config
        self.max_context = teacher_config.max_context_chars

    # ------------------------------------------------------------------
    # 프롬프트 빌드
    # ------------------------------------------------------------------

    def _build_extraction_prompt(self, doc_title: str, content: str) -> str:
        """엔티티·관계 추출을 위한 프롬프트를 구성합니다."""
        truncated = content[: self.max_context]
        if len(content) > self.max_context:
            truncated += "\n... (이하 생략)"

        entity_types = ", ".join(self.config.entity_types)

        return (
            "다음 문서에서 핵심 엔티티(개체)와 엔티티 간 관계를 추출해주세요.\n\n"
            "## 규칙\n"
            f"- 엔티티 유형은 다음 중에서만 선택: {entity_types}\n"
            "- 관계의 주어(subject)와 목적어(object)는 반드시 추출한 엔티티 이름이어야 합니다\n"
            "- 엔티티 이름은 문서에 등장하는 정확한 명칭을 사용하세요\n"
            "- 확신도(confidence)는 0.0~1.0 사이 실수로 표현하세요\n"
            "- 관계의 술어(predicate)는 간결한 동사/명사형으로 표현하세요 "
            '(예: "소속", "개발", "위치", "포함")\n'
            "- 중요하지 않은 일반 명사는 엔티티로 추출하지 마세요\n\n"
            f"## 문서 제목\n{doc_title}\n\n"
            f"## 문서 내용\n{truncated}\n\n"
            "## 출력 형식\n"
            "반드시 아래 JSON 형식으로만 응답하세요:\n"
            "```json\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "엔티티명", "entity_type": "유형", "confidence": 0.9}\n'
            "  ],\n"
            '  "relations": [\n'
            '    {"subject": "주어", "predicate": "술어", "object": "목적어", '
            '"confidence": 0.8}\n'
            "  ]\n"
            "}\n"
            "```"
        )

    # ------------------------------------------------------------------
    # 파싱
    # ------------------------------------------------------------------

    def _parse_extraction(
        self, text: str, source_doc: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """LLM 응답에서 엔티티와 관계를 추출합니다."""
        text = text.strip()
        # 코드 블록 마커 제거
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        data: dict[str, Any] = {}

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.debug("온톨로지 JSON 파싱 실패, 정규식 fallback 시도: %s", text[:80])
            # JSON 블록 추출 시도
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    logger.warning("온톨로지 응답 파싱 최종 실패: %s", text[:100])
                    return [], []
            else:
                logger.warning("온톨로지 응답에서 JSON을 찾을 수 없음: %s", text[:100])
                return [], []

        entities: list[Entity] = []
        for raw in data.get("entities", []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            entity_type = str(raw.get("entity_type", "")).strip()
            if not name or not entity_type:
                continue
            entities.append(Entity(
                name=name,
                entity_type=entity_type,
                source_doc=source_doc,
                confidence=float(raw.get("confidence", 1.0)),
            ))

        relations: list[Relation] = []
        for raw in data.get("relations", []):
            if not isinstance(raw, dict):
                continue
            subject = str(raw.get("subject", "")).strip()
            predicate = str(raw.get("predicate", "")).strip()
            obj = str(raw.get("object", "")).strip()
            if not subject or not predicate or not obj:
                continue
            relations.append(Relation(
                subject=subject,
                predicate=predicate,
                object=obj,
                source_doc=source_doc,
                confidence=float(raw.get("confidence", 1.0)),
            ))

        return entities, relations

    # ------------------------------------------------------------------
    # 검증
    # ------------------------------------------------------------------

    def _validate_extraction(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[list[Entity], list[Relation]]:
        """추출 결과를 검증하고 허용되지 않는 항목을 제거합니다.

        - 엔티티: 허용된 유형 목록에 포함되지 않으면 제거
        - 엔티티: 최소 확신도 미만이면 제거
        - 관계: 주어·목적어가 남은 엔티티에 없으면 제거
        - 관계: 최소 확신도 미만이면 제거
        """
        allowed_types = {t.strip() for t in self.config.entity_types}
        min_conf = self.config.min_confidence

        # 엔티티 필터링
        valid_entities = [
            e for e in entities
            if e.entity_type in allowed_types and e.confidence >= min_conf
        ]

        removed_entity_count = len(entities) - len(valid_entities)
        if removed_entity_count:
            logger.debug(
                "엔티티 검증: %d개 중 %d개 제거 (유형/확신도 미달)",
                len(entities), removed_entity_count,
            )

        # 관계 필터링: 주어·목적어가 유효한 엔티티 이름에 포함되는지 확인
        entity_names = {e.name for e in valid_entities}
        valid_relations = [
            r for r in relations
            if (
                r.subject in entity_names
                and r.object in entity_names
                and r.confidence >= min_conf
            )
        ]

        removed_rel_count = len(relations) - len(valid_relations)
        if removed_rel_count:
            logger.debug(
                "관계 검증: %d개 중 %d개 제거 (참조 엔티티 없음/확신도 미달)",
                len(relations), removed_rel_count,
            )

        return valid_entities, valid_relations

    # ------------------------------------------------------------------
    # 정규화
    # ------------------------------------------------------------------

    def _normalize_entities(self, entities: list[Entity]) -> list[Entity]:
        """동일 엔티티의 이름을 통일합니다.

        ``(name.upper(), entity_type)`` 기준으로 동일 엔티티를 그룹화하고,
        더 긴 정규 이름(canonical)을 선택합니다. 확신도는 그룹 내 최댓값을
        사용합니다.
        """
        # 키: (name.upper(), entity_type) → 대표 엔티티
        canonical: dict[tuple[str, str], Entity] = {}

        for entity in entities:
            key = (entity.name.upper(), entity.entity_type)
            existing = canonical.get(key)
            if existing is None:
                canonical[key] = entity
            else:
                # 더 긴 이름을 정규 이름으로 선택
                if len(entity.name) > len(existing.name):
                    canonical[key] = Entity(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        source_doc=existing.source_doc,
                        confidence=max(entity.confidence, existing.confidence),
                        properties=existing.properties,
                    )
                else:
                    canonical[key] = Entity(
                        name=existing.name,
                        entity_type=existing.entity_type,
                        source_doc=existing.source_doc,
                        confidence=max(entity.confidence, existing.confidence),
                        properties=existing.properties,
                    )

        return list(canonical.values())

    # ------------------------------------------------------------------
    # 비동기 단건/일괄 처리
    # ------------------------------------------------------------------

    async def _extract_chunk(
        self, doc_title: str, content: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """단일 청크에서 엔티티와 관계를 추출합니다."""
        prompt = self._build_extraction_prompt(doc_title, content)

        kwargs: dict[str, Any] = {}
        if self.teacher_config.backend == "ollama":
            kwargs["format"] = "json"

        response = await self.teacher.agenerate(prompt, **kwargs)
        entities, relations = self._parse_extraction(response, doc_title)
        return self._validate_extraction(entities, relations)

    async def extract_one(
        self, doc: ParsedDocument,
    ) -> tuple[list[Entity], list[Relation]]:
        """단일 문서에서 엔티티와 관계를 추출합니다.

        문서가 max_context_chars보다 길면 청크로 분할하여 각 청크에서
        추출한 뒤 결과를 병합합니다. 이를 통해 전체 문서의 지식을 포착합니다.
        """
        chunks = chunk_document(
            doc.content, self.max_context, self.max_context // 4,
        )

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for i, chunk in enumerate(chunks):
            try:
                entities, relations = await self._extract_chunk(doc.title, chunk)
                all_entities.extend(entities)
                all_relations.extend(relations)
            except Exception as e:
                logger.warning(
                    "문서 '%s' 청크 %d/%d 추출 실패: %s",
                    doc.title, i + 1, len(chunks), e,
                )

        all_entities = self._normalize_entities(all_entities)

        logger.debug(
            "문서 '%s': %d 청크 → %d 엔티티, %d 관계 추출",
            doc.title, len(chunks), len(all_entities), len(all_relations),
        )
        return all_entities, all_relations

    async def extract_all(
        self, docs: list[ParsedDocument],
    ) -> KnowledgeGraph:
        """전체 문서에서 온톨로지를 추출합니다.

        ``config.max_concurrency``에 따라 동시 요청 수를 제한합니다.
        추출 후 엔티티 정규화를 수행합니다.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        with progress:
            task_id = progress.add_task(
                "온톨로지 추출 중...", total=len(docs),
            )

            tasks = [
                run_bounded(semaphore, self.extract_one(doc), progress, task_id)
                for doc in docs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(
                    "문서 '%s' 온톨로지 추출 실패: %s",
                    docs[i].title, result,
                )
                continue
            entities, relations = result
            all_entities.extend(entities)
            all_relations.extend(relations)

        # 엔티티 정규화
        all_entities = self._normalize_entities(all_entities)

        logger.info(
            "온톨로지 추출 완료: %d개 문서 → %d 엔티티, %d 관계",
            len(docs), len(all_entities), len(all_relations),
        )
        return KnowledgeGraph(entities=all_entities, relations=all_relations)
