"""온톨로지 지식 그래프 저장소(GraphStore)의 단위 테스트입니다."""

from __future__ import annotations

import json

import pytest

from slm_factory.ontology.graph_store import GraphStore
from slm_factory.ontology.models import Entity, KnowledgeGraph, Relation


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_entity(
    name: str,
    entity_type: str = "Concept",
    source_doc: str = "doc1.pdf",
    confidence: float = 1.0,
) -> Entity:
    return Entity(
        name=name,
        entity_type=entity_type,
        source_doc=source_doc,
        confidence=confidence,
    )


def _make_relation(
    subject: str,
    object_: str,
    predicate: str = "관련",
    source_doc: str = "doc1.pdf",
    confidence: float = 1.0,
) -> Relation:
    return Relation(
        subject=subject,
        predicate=predicate,
        object=object_,
        source_doc=source_doc,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 엔티티 직렬화
# ---------------------------------------------------------------------------


class TestEntitySerialization:
    """Entity ↔ dict 변환 테스트입니다."""

    def test_엔티티_딕셔너리_변환(self):
        """Entity를 딕셔너리로 변환하면 모든 필드가 포함됩니다."""
        e = Entity(
            name="Python",
            entity_type="Technology",
            source_doc="doc.pdf",
            confidence=0.9,
            properties={"desc": "프로그래밍 언어"},
        )
        d = GraphStore._entity_to_dict(e)
        assert d["name"] == "Python"
        assert d["entity_type"] == "Technology"
        assert d["source_doc"] == "doc.pdf"
        assert d["confidence"] == 0.9
        assert d["properties"]["desc"] == "프로그래밍 언어"

    def test_딕셔너리_엔티티_복원(self):
        """딕셔너리에서 Entity를 복원합니다."""
        d = {
            "name": "Python",
            "entity_type": "Technology",
            "source_doc": "doc.pdf",
            "confidence": 0.9,
            "properties": {"desc": "언어"},
        }
        e = GraphStore._entity_from_dict(d)
        assert e.name == "Python"
        assert e.entity_type == "Technology"
        assert e.confidence == 0.9

    def test_엔티티_라운드트립(self):
        """Entity → dict → Entity 라운드트립에서 데이터가 보존됩니다."""
        original = Entity(name="테스트", entity_type="Concept", confidence=0.75)
        restored = GraphStore._entity_from_dict(GraphStore._entity_to_dict(original))
        assert restored.name == original.name
        assert restored.entity_type == original.entity_type
        assert restored.confidence == original.confidence

    def test_선택_필드_기본값_복원(self):
        """선택 필드가 없는 딕셔너리에서 기본값으로 복원됩니다."""
        d = {"name": "X"}
        e = GraphStore._entity_from_dict(d)
        assert e.entity_type == ""
        assert e.source_doc == ""
        assert e.confidence == 1.0
        assert e.properties == {}


# ---------------------------------------------------------------------------
# 관계 직렬화
# ---------------------------------------------------------------------------


class TestRelationSerialization:
    """Relation ↔ dict 변환 테스트입니다."""

    def test_관계_딕셔너리_변환(self):
        """Relation을 딕셔너리로 변환하면 모든 필드가 포함됩니다."""
        r = Relation(
            subject="A", predicate="포함", object="B",
            source_doc="doc.pdf", confidence=0.8,
        )
        d = GraphStore._relation_to_dict(r)
        assert d["subject"] == "A"
        assert d["predicate"] == "포함"
        assert d["object"] == "B"
        assert d["confidence"] == 0.8

    def test_딕셔너리_관계_복원(self):
        """딕셔너리에서 Relation을 복원합니다."""
        d = {"subject": "A", "predicate": "포함", "object": "B", "confidence": 0.8}
        r = GraphStore._relation_from_dict(d)
        assert r.subject == "A"
        assert r.predicate == "포함"
        assert r.object == "B"
        assert r.confidence == 0.8

    def test_관계_라운드트립(self):
        """Relation → dict → Relation 라운드트립에서 데이터가 보존됩니다."""
        original = Relation(
            subject="X", predicate="개발", object="Y", confidence=0.6,
        )
        restored = GraphStore._relation_from_dict(
            GraphStore._relation_to_dict(original),
        )
        assert restored.subject == original.subject
        assert restored.predicate == original.predicate
        assert restored.confidence == original.confidence

    def test_선택_필드_기본값_복원(self):
        """선택 필드가 없는 딕셔너리에서 기본값으로 복원됩니다."""
        d = {"subject": "A", "object": "B"}
        r = GraphStore._relation_from_dict(d)
        assert r.predicate == ""
        assert r.source_doc == ""
        assert r.confidence == 1.0


# ---------------------------------------------------------------------------
# 저장 (save)
# ---------------------------------------------------------------------------


class TestSave:
    """GraphStore.save() 테스트입니다."""

    def test_정상_저장(self, tmp_path):
        """지식 그래프를 JSON 파일로 저장합니다."""
        kg = KnowledgeGraph(
            entities=[_make_entity("A")],
            relations=[_make_relation("A", "B")],
        )
        path = tmp_path / "kg.json"
        GraphStore.save(kg, path)

        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["entities"]) == 1
        assert len(data["relations"]) == 1

    def test_부모_디렉토리_자동_생성(self, tmp_path):
        """부모 디렉토리가 없으면 자동으로 생성합니다."""
        path = tmp_path / "sub" / "dir" / "kg.json"
        GraphStore.save(KnowledgeGraph(), path)
        assert path.is_file()

    def test_한글_인코딩_보존(self, tmp_path):
        """ensure_ascii=False로 한글이 그대로 저장됩니다."""
        kg = KnowledgeGraph(
            entities=[_make_entity("삼성전자", entity_type="Organization")],
        )
        path = tmp_path / "kg.json"
        GraphStore.save(kg, path)

        text = path.read_text(encoding="utf-8")
        assert "삼성전자" in text
        assert "\\u" not in text

    def test_빈_그래프_저장(self, tmp_path):
        """빈 그래프도 정상적으로 저장됩니다."""
        path = tmp_path / "empty.json"
        GraphStore.save(KnowledgeGraph(), path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["entities"] == []
        assert data["relations"] == []


# ---------------------------------------------------------------------------
# 로드 (load)
# ---------------------------------------------------------------------------


class TestLoad:
    """GraphStore.load() 테스트입니다."""

    def test_정상_로드(self, tmp_path):
        """JSON 파일에서 지식 그래프를 로드합니다."""
        path = tmp_path / "kg.json"
        data = {
            "entities": [{"name": "Python", "entity_type": "Technology"}],
            "relations": [
                {"subject": "Python", "predicate": "사용", "object": "개발자"},
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        kg = GraphStore.load(path)
        assert len(kg.entities) == 1
        assert kg.entities[0].name == "Python"
        assert len(kg.relations) == 1
        assert kg.relations[0].predicate == "사용"

    def test_파일_없으면_빈_그래프(self, tmp_path):
        """존재하지 않는 파일을 로드하면 빈 그래프를 반환합니다."""
        kg = GraphStore.load(tmp_path / "nonexistent.json")
        assert kg.entities == []
        assert kg.relations == []

    def test_손상된_파일_빈_그래프(self, tmp_path):
        """손상된 JSON 파일을 로드하면 빈 그래프를 반환합니다."""
        path = tmp_path / "broken.json"
        path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        kg = GraphStore.load(path)
        assert kg.entities == []
        assert kg.relations == []

    def test_저장_후_로드_일치(self, tmp_path):
        """save → load 라운드트립에서 데이터가 보존됩니다."""
        original = KnowledgeGraph(
            entities=[
                _make_entity("A", entity_type="Person", confidence=0.9),
                _make_entity("B", entity_type="Organization"),
            ],
            relations=[
                _make_relation("A", "B", predicate="소속", confidence=0.8),
            ],
        )
        path = tmp_path / "kg.json"
        GraphStore.save(original, path)
        loaded = GraphStore.load(path)

        assert len(loaded.entities) == len(original.entities)
        assert len(loaded.relations) == len(original.relations)
        assert loaded.entities[0].name == "A"
        assert loaded.relations[0].predicate == "소속"
        assert loaded.relations[0].confidence == 0.8


# ---------------------------------------------------------------------------
# 병합 (merge)
# ---------------------------------------------------------------------------


class TestMerge:
    """GraphStore.merge() 테스트입니다."""

    def test_변경_문서_교체(self):
        """changed_docs에 해당하는 기존 항목이 새 항목으로 교체됩니다."""
        existing = KnowledgeGraph(
            entities=[_make_entity("Old", source_doc="doc1.pdf")],
            relations=[_make_relation("Old", "X", source_doc="doc1.pdf")],
        )
        new = KnowledgeGraph(
            entities=[_make_entity("New", source_doc="doc1.pdf")],
            relations=[_make_relation("New", "Y", source_doc="doc1.pdf")],
        )
        merged = GraphStore.merge(
            existing, new, changed_docs={"doc1.pdf"}, deleted_docs=set(),
        )

        names = [e.name for e in merged.entities]
        assert "New" in names
        assert "Old" not in names

    def test_삭제_문서_제거(self):
        """deleted_docs에 해당하는 엔티티·관계가 제거됩니다."""
        existing = KnowledgeGraph(
            entities=[
                _make_entity("Keep", source_doc="doc1.pdf"),
                _make_entity("Remove", source_doc="doc2.pdf"),
            ],
            relations=[
                _make_relation("Keep", "X", source_doc="doc1.pdf"),
                _make_relation("Remove", "Y", source_doc="doc2.pdf"),
            ],
        )
        merged = GraphStore.merge(
            existing, KnowledgeGraph(),
            changed_docs=set(), deleted_docs={"doc2.pdf"},
        )

        names = [e.name for e in merged.entities]
        assert "Keep" in names
        assert "Remove" not in names
        assert len(merged.relations) == 1

    def test_변경_없는_문서_유지(self):
        """changed_docs·deleted_docs에 포함되지 않은 항목은 유지됩니다."""
        existing = KnowledgeGraph(
            entities=[_make_entity("Unchanged", source_doc="doc1.pdf")],
        )
        new = KnowledgeGraph(
            entities=[_make_entity("Added", source_doc="doc2.pdf")],
        )
        merged = GraphStore.merge(
            existing, new, changed_docs={"doc2.pdf"}, deleted_docs=set(),
        )

        names = [e.name for e in merged.entities]
        assert "Unchanged" in names
        assert "Added" in names

    def test_엔티티_중복_제거(self):
        """(name.upper(), entity_type, source_doc) 기준으로 중복 제거합니다."""
        existing = KnowledgeGraph(
            entities=[
                _make_entity("A", source_doc="doc1.pdf"),
                _make_entity("A", source_doc="doc1.pdf"),
            ],
        )
        merged = GraphStore.merge(
            existing, KnowledgeGraph(), changed_docs=set(), deleted_docs=set(),
        )

        names = [e.name for e in merged.entities]
        assert names.count("A") == 1

    def test_관계_중복_제거(self):
        """(subject, predicate, object, source_doc) 기준으로 중복 제거합니다."""
        existing = KnowledgeGraph(
            relations=[
                _make_relation("A", "B", source_doc="doc1.pdf"),
                _make_relation("A", "B", source_doc="doc1.pdf"),
            ],
        )
        merged = GraphStore.merge(
            existing, KnowledgeGraph(), changed_docs=set(), deleted_docs=set(),
        )
        assert len(merged.relations) == 1

    def test_빈_그래프_병합(self):
        """빈 그래프끼리 병합하면 빈 그래프를 반환합니다."""
        merged = GraphStore.merge(
            KnowledgeGraph(), KnowledgeGraph(),
            changed_docs=set(), deleted_docs=set(),
        )
        assert merged.entities == []
        assert merged.relations == []

    def test_복합_시나리오(self):
        """신규·수정·삭제·유지가 동시에 발생하는 시나리오를 검증합니다."""
        existing = KnowledgeGraph(
            entities=[
                _make_entity("Keep", source_doc="doc1.pdf"),
                _make_entity("Update", source_doc="doc2.pdf"),
                _make_entity("Delete", source_doc="doc3.pdf"),
            ],
        )
        new = KnowledgeGraph(
            entities=[
                _make_entity("Updated", source_doc="doc2.pdf"),
                _make_entity("Brand_New", source_doc="doc4.pdf"),
            ],
        )
        merged = GraphStore.merge(
            existing, new,
            changed_docs={"doc2.pdf", "doc4.pdf"},
            deleted_docs={"doc3.pdf"},
        )

        names = [e.name for e in merged.entities]
        assert "Keep" in names
        assert "Updated" in names
        assert "Brand_New" in names
        assert "Update" not in names
        assert "Delete" not in names
