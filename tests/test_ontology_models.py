"""온톨로지 데이터 모델(Entity, Relation, KnowledgeGraph)의 단위 테스트입니다."""

from __future__ import annotations

from slm_factory.ontology.models import Entity, KnowledgeGraph, Relation


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class TestEntity:
    """Entity 데이터클래스의 테스트입니다."""

    def test_기본_생성_및_필드_검증(self):
        """Entity를 필수 필드만으로 생성하고 기본값을 확인합니다."""
        e = Entity(name="삼성전자", entity_type="Organization")
        assert e.name == "삼성전자"
        assert e.entity_type == "Organization"
        assert e.source_doc == ""
        assert e.confidence == 1.0
        assert e.properties == {}

    def test_커스텀_필드_지정(self):
        """모든 필드를 명시적으로 지정하여 올바르게 저장되는지 확인합니다."""
        e = Entity(
            name="홍길동",
            entity_type="Person",
            source_doc="doc1.pdf",
            confidence=0.85,
            properties={"alias": "길동이"},
        )
        assert e.name == "홍길동"
        assert e.entity_type == "Person"
        assert e.source_doc == "doc1.pdf"
        assert e.confidence == 0.85
        assert e.properties["alias"] == "길동이"

    def test_기본값_독립성(self):
        """서로 다른 인스턴스의 properties가 공유되지 않는지 확인합니다."""
        e1 = Entity(name="A", entity_type="Concept")
        e2 = Entity(name="B", entity_type="Concept")
        e1.properties["key"] = "val"
        assert e2.properties == {}


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


class TestRelation:
    """Relation 데이터클래스의 테스트입니다."""

    def test_기본_생성_및_필드_검증(self):
        """Relation을 필수 필드만으로 생성하고 기본값을 확인합니다."""
        r = Relation(subject="삼성전자", predicate="개발", object="갤럭시")
        assert r.subject == "삼성전자"
        assert r.predicate == "개발"
        assert r.object == "갤럭시"
        assert r.source_doc == ""
        assert r.confidence == 1.0

    def test_커스텀_필드_지정(self):
        """모든 필드를 명시적으로 지정하여 올바르게 저장되는지 확인합니다."""
        r = Relation(
            subject="A",
            predicate="포함",
            object="B",
            source_doc="doc.pdf",
            confidence=0.7,
        )
        assert r.source_doc == "doc.pdf"
        assert r.confidence == 0.7


# ---------------------------------------------------------------------------
# KnowledgeGraph — 기본
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    """KnowledgeGraph 데이터클래스의 기본 테스트입니다."""

    def test_빈_그래프_생성(self):
        """기본 생성 시 빈 엔티티·관계 리스트를 가집니다."""
        kg = KnowledgeGraph()
        assert kg.entities == []
        assert kg.relations == []

    def test_기본값_독립성(self):
        """서로 다른 인스턴스의 리스트가 공유되지 않는지 확인합니다."""
        kg1 = KnowledgeGraph()
        kg2 = KnowledgeGraph()
        kg1.entities.append(Entity(name="X", entity_type="Concept"))
        assert kg2.entities == []

    def test_엔티티와_관계_직접_지정(self):
        """엔티티·관계를 생성자에 직접 전달하여 저장되는지 확인합니다."""
        entities = [Entity(name="A", entity_type="Person")]
        relations = [Relation(subject="A", predicate="소속", object="B")]
        kg = KnowledgeGraph(entities=entities, relations=relations)
        assert len(kg.entities) == 1
        assert len(kg.relations) == 1


# ---------------------------------------------------------------------------
# KnowledgeGraph.to_context_string
# ---------------------------------------------------------------------------


class TestToContextString:
    """to_context_string() 메서드의 테스트입니다."""

    def test_빈_그래프_빈_문자열(self):
        """엔티티·관계가 없으면 빈 문자열을 반환합니다."""
        kg = KnowledgeGraph()
        assert kg.to_context_string() == ""

    def test_엔티티만_있는_경우(self):
        """엔티티만 있을 때 엔티티 정보를 포함합니다."""
        kg = KnowledgeGraph(
            entities=[Entity(name="Python", entity_type="Technology")],
        )
        result = kg.to_context_string()
        assert "Python" in result
        assert "Technology" in result

    def test_source_doc_필터링(self):
        """source_doc를 지정하면 해당 문서의 엔티티만 포함합니다."""
        kg = KnowledgeGraph(
            entities=[
                Entity(name="A", entity_type="Concept", source_doc="doc1.pdf"),
                Entity(name="B", entity_type="Concept", source_doc="doc2.pdf"),
            ],
        )
        result = kg.to_context_string(source_doc="doc1.pdf")
        assert "A" in result
        assert "B" not in result

    def test_source_doc_필터링_후_빈_결과(self):
        """source_doc 필터링 결과가 없으면 빈 문자열을 반환합니다."""
        kg = KnowledgeGraph(
            entities=[Entity(name="A", entity_type="Concept", source_doc="doc1.pdf")],
        )
        assert kg.to_context_string(source_doc="없는문서.pdf") == ""

    def test_max_items_제한(self):
        """max_items로 포함할 엔티티 수를 제한합니다."""
        entities = [
            Entity(name=f"E{i}", entity_type="Concept", confidence=i / 10)
            for i in range(10)
        ]
        kg = KnowledgeGraph(entities=entities)
        result = kg.to_context_string(max_items=3)
        entity_lines = [l for l in result.split("\n") if l.startswith("- Entity:")]
        assert len(entity_lines) == 3

    def test_확신도_내림차순_정렬(self):
        """높은 확신도의 엔티티가 우선 포함됩니다."""
        entities = [
            Entity(name="Low", entity_type="Concept", confidence=0.1),
            Entity(name="High", entity_type="Concept", confidence=0.9),
            Entity(name="Mid", entity_type="Concept", confidence=0.5),
        ]
        kg = KnowledgeGraph(entities=entities)
        result = kg.to_context_string(max_items=2)
        assert "High" in result
        assert "Mid" in result
        assert "Low" not in result

    def test_관계_포함(self):
        """엔티티에 연결된 관계가 출력에 포함됩니다."""
        entities = [
            Entity(name="A", entity_type="Concept"),
            Entity(name="B", entity_type="Concept"),
        ]
        relations = [Relation(subject="A", predicate="포함", object="B")]
        kg = KnowledgeGraph(entities=entities, relations=relations)
        result = kg.to_context_string()
        assert "A → 포함 → B" in result

    def test_관계_참조_엔티티_제외시_관계도_제외(self):
        """max_items로 엔티티가 잘리면 해당 관계도 제외됩니다."""
        entities = [
            Entity(name="A", entity_type="Concept", confidence=0.9),
            Entity(name="B", entity_type="Concept", confidence=0.1),
        ]
        relations = [Relation(subject="A", predicate="관련", object="B")]
        kg = KnowledgeGraph(entities=entities, relations=relations)
        result = kg.to_context_string(max_items=1)
        assert "A" in result
        assert "관련" not in result


# ---------------------------------------------------------------------------
# KnowledgeGraph.export_triples
# ---------------------------------------------------------------------------


class TestExportTriples:
    """export_triples() 메서드의 테스트입니다."""

    def test_빈_관계_빈_리스트(self):
        """관계가 없으면 빈 리스트를 반환합니다."""
        kg = KnowledgeGraph()
        assert kg.export_triples() == []

    def test_SPO_트리플_변환(self):
        """관계를 (subject, predicate, object) 튜플로 변환합니다."""
        kg = KnowledgeGraph(
            relations=[
                Relation(subject="A", predicate="소속", object="B"),
                Relation(subject="C", predicate="개발", object="D"),
            ],
        )
        triples = kg.export_triples()
        assert ("A", "소속", "B") in triples
        assert ("C", "개발", "D") in triples
        assert len(triples) == 2
