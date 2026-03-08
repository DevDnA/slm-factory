"""온톨로지 지식 그래프 데이터 모델.

문서에서 추출한 엔티티(개체)와 관계를 표현하는 데이터클래스를 정의합니다.
QA 생성 시 컨텍스트로 주입하거나 독립적으로 저장/조회할 수 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """지식 그래프의 엔티티(개체)입니다.

    문서에서 추출된 명명된 개체를 나타냅니다.
    동일 엔티티는 ``(name.upper(), entity_type)`` 조합으로 식별합니다.
    """

    name: str
    """엔티티의 정규 이름입니다 (예: "삼성전자")."""

    entity_type: str
    """엔티티 유형입니다 (예: "Organization", "Person", "Concept")."""

    source_doc: str = ""
    """이 엔티티를 추출한 원본 문서 ID입니다."""

    confidence: float = 1.0
    """추출 확신도입니다 (0.0~1.0)."""

    properties: dict[str, Any] = field(default_factory=dict)
    """엔티티의 추가 속성입니다 (설명, 별칭 등)."""


@dataclass
class Relation:
    """엔티티 간의 관계(트리플)입니다.

    주어-술어-목적어 형태의 지식 트리플을 나타냅니다.
    중복 판단 키는 ``(subject, predicate, object, source_doc)``입니다.
    """

    subject: str
    """주어 엔티티 이름입니다."""

    predicate: str
    """관계 유형입니다 (예: "소속", "개발", "위치")."""

    object: str
    """목적어 엔티티 이름입니다."""

    source_doc: str = ""
    """이 관계를 추출한 원본 문서 ID입니다."""

    confidence: float = 1.0
    """추출 확신도입니다 (0.0~1.0)."""


@dataclass
class KnowledgeGraph:
    """문서에서 추출된 지식 그래프입니다.

    엔티티와 관계의 집합으로 구성됩니다.
    QA 생성 컨텍스트 포맷팅, 외부 시스템 연동용 트리플 내보내기를
    지원합니다.
    """

    entities: list[Entity] = field(default_factory=list)
    """추출된 엔티티 목록입니다."""

    relations: list[Relation] = field(default_factory=list)
    """추출된 관계 목록입니다."""

    def to_context_string(
        self,
        source_doc: str | None = None,
        max_items: int = 20,
    ) -> str:
        """QA 프롬프트 주입용 컨텍스트 문자열을 생성합니다.

        매개변수
        ----------
        source_doc:
            특정 문서의 엔티티/관계만 필터링합니다.
            ``None``이면 전체를 대상으로 합니다.
        max_items:
            포함할 최대 엔티티 수입니다. 확신도 내림차순으로
            상위 항목을 선택합니다.

        반환값
        -------
        str
            프롬프트에 삽입할 수 있는 포맷된 문자열입니다.
            엔티티나 관계가 없으면 빈 문자열을 반환합니다.
        """
        entities = self.entities
        relations = self.relations

        if source_doc:
            entities = [e for e in entities if e.source_doc == source_doc]
            relations = [r for r in relations if r.source_doc == source_doc]

        if not entities and not relations:
            return ""

        # 확신도 내림차순으로 상위 N개 엔티티 선택
        entities = sorted(
            entities, key=lambda e: e.confidence, reverse=True,
        )[:max_items]
        entity_names = {e.name for e in entities}

        # 포함된 엔티티의 관계만 선택
        relations = [
            r for r in relations
            if r.subject in entity_names and r.object in entity_names
        ]

        lines: list[str] = []
        for e in entities:
            lines.append(f"- Entity: {e.name} ({e.entity_type})")
        for r in relations:
            lines.append(f"- Relation: {r.subject} → {r.predicate} → {r.object}")

        return "\n".join(lines)

    def export_triples(self) -> list[tuple[str, str, str]]:
        """외부 시스템(RAG 등) 연동용 SPO 트리플 리스트를 반환합니다.

        반환값
        -------
        list[tuple[str, str, str]]
            ``(subject, predicate, object)`` 튜플의 리스트입니다.
        """
        return [(r.subject, r.predicate, r.object) for r in self.relations]
