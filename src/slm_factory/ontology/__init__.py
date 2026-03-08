"""온톨로지/지식 그래프 모듈.

문서에서 엔티티(개체)와 관계를 추출하여 지식 그래프를 구성합니다.
독립 실행 또는 QA 생성 시 컨텍스트 강화에 활용할 수 있습니다.
"""

from __future__ import annotations

from .extractor import OntologyExtractor
from .graph_store import GraphStore
from .models import Entity, KnowledgeGraph, Relation

__all__ = [
    "Entity",
    "GraphStore",
    "KnowledgeGraph",
    "OntologyExtractor",
    "Relation",
]
