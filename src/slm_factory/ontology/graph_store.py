"""온톨로지 지식 그래프의 JSON 파일 저장소입니다.

지식 그래프를 JSON 형식으로 직렬화/역직렬화하고,
증분 업데이트 시 기존 그래프와 새 그래프를 병합합니다.
삭제된 문서의 엔티티·관계도 올바르게 제거합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from ..utils import get_logger
from .models import Entity, KnowledgeGraph, Relation

logger = get_logger("ontology.graph_store")


class GraphStore:
    """지식 그래프의 JSON 파일 직렬화를 담당합니다."""

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_to_dict(entity: Entity) -> dict[str, Any]:
        """엔티티를 직렬화 가능한 딕셔너리로 변환합니다."""
        return {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "source_doc": entity.source_doc,
            "confidence": entity.confidence,
            "properties": entity.properties,
        }

    @staticmethod
    def _relation_to_dict(relation: Relation) -> dict[str, Any]:
        """관계를 직렬화 가능한 딕셔너리로 변환합니다."""
        return {
            "subject": relation.subject,
            "predicate": relation.predicate,
            "object": relation.object,
            "source_doc": relation.source_doc,
            "confidence": relation.confidence,
        }

    @staticmethod
    def _entity_from_dict(data: dict[str, Any]) -> Entity:
        """딕셔너리에서 엔티티를 복원합니다."""
        return Entity(
            name=data["name"],
            entity_type=data.get("entity_type", ""),
            source_doc=data.get("source_doc", ""),
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
        )

    @staticmethod
    def _relation_from_dict(data: dict[str, Any]) -> Relation:
        """딕셔너리에서 관계를 복원합니다."""
        return Relation(
            subject=data["subject"],
            predicate=data.get("predicate", ""),
            object=data["object"],
            source_doc=data.get("source_doc", ""),
            confidence=data.get("confidence", 1.0),
        )

    # ------------------------------------------------------------------
    # 파일 I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save(kg: KnowledgeGraph, path: Path) -> None:
        """지식 그래프를 JSON 파일로 저장합니다.

        매개변수
        ----------
        kg:
            저장할 지식 그래프 객체입니다.
        path:
            저장 경로입니다. 부모 디렉토리가 없으면 자동 생성합니다.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entities": [GraphStore._entity_to_dict(e) for e in kg.entities],
            "relations": [GraphStore._relation_to_dict(r) for r in kg.relations],
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "지식 그래프 저장: %d 엔티티, %d 관계 → %s",
            len(kg.entities), len(kg.relations), path,
        )

    @staticmethod
    def load(path: Path) -> KnowledgeGraph:
        """JSON 파일에서 지식 그래프를 로드합니다.

        매개변수
        ----------
        path:
            로드할 JSON 파일 경로입니다.

        반환값
        -------
        KnowledgeGraph
            복원된 지식 그래프입니다. 파일이 없으면 빈 그래프를 반환합니다.
        """
        path = Path(path)
        if not path.is_file():
            return KnowledgeGraph()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "온톨로지 파일이 손상되어 빈 그래프로 초기화합니다: %s", path,
            )
            return KnowledgeGraph()

        entities = [
            GraphStore._entity_from_dict(d) for d in raw.get("entities", [])
        ]
        relations = [
            GraphStore._relation_from_dict(d) for d in raw.get("relations", [])
        ]

        logger.info(
            "지식 그래프 로드: %d 엔티티, %d 관계 ← %s",
            len(entities), len(relations), path,
        )
        return KnowledgeGraph(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # 병합
    # ------------------------------------------------------------------

    @staticmethod
    def merge(
        existing: KnowledgeGraph,
        new: KnowledgeGraph,
        changed_docs: set[str],
        deleted_docs: set[str],
    ) -> KnowledgeGraph:
        """기존 그래프와 새 그래프를 병합합니다.

        증분 업데이트 시나리오에서 사용합니다:

        1. ``deleted_docs``에 해당하는 엔티티·관계를 기존 그래프에서 제거합니다.
        2. ``changed_docs``에 해당하는 기존 엔티티·관계를 교체합니다
           (새 그래프의 엔티티·관계로 대체).
        3. 변경되지 않은 문서의 엔티티·관계는 그대로 유지합니다.
        4. 최종 중복 제거를 수행합니다.

        매개변수
        ----------
        existing:
            기존 저장된 지식 그래프입니다.
        new:
            이번에 새로 추출된 지식 그래프입니다.
        changed_docs:
            변경(신규 + 수정)된 문서 ID 집합입니다.
        deleted_docs:
            삭제된 문서 ID 집합입니다.

        반환값
        -------
        KnowledgeGraph
            병합된 지식 그래프입니다.
        """
        remove_docs = changed_docs | deleted_docs

        # 기존 그래프에서 변경·삭제 대상 문서의 항목 제거
        kept_entities = [
            e for e in existing.entities if e.source_doc not in remove_docs
        ]
        kept_relations = [
            r for r in existing.relations if r.source_doc not in remove_docs
        ]

        # 새 항목 추가
        merged_entities = kept_entities + list(new.entities)
        merged_relations = kept_relations + list(new.relations)

        # 엔티티 중복 제거: (name.upper(), entity_type, source_doc)
        seen_entities: set[tuple[str, str, str]] = set()
        deduped_entities: list[Entity] = []
        for entity in merged_entities:
            key = (entity.name.upper(), entity.entity_type, entity.source_doc)
            if key not in seen_entities:
                seen_entities.add(key)
                deduped_entities.append(entity)

        # 관계 중복 제거: (subject, predicate, object, source_doc)
        seen_relations: set[tuple[str, str, str, str]] = set()
        deduped_relations: list[Relation] = []
        for relation in merged_relations:
            key = (
                relation.subject,
                relation.predicate,
                relation.object,
                relation.source_doc,
            )
            if key not in seen_relations:
                seen_relations.add(key)
                deduped_relations.append(relation)

        logger.info(
            "지식 그래프 병합: 기존 %d+%d → 병합 %d+%d (변경 %d, 삭제 %d 문서)",
            len(existing.entities), len(existing.relations),
            len(deduped_entities), len(deduped_relations),
            len(changed_docs), len(deleted_docs),
        )
        return KnowledgeGraph(
            entities=deduped_entities,
            relations=deduped_relations,
        )
