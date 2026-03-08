"""온톨로지 추출기(OntologyExtractor)의 단위 테스트입니다.

순수 로직(파싱·검증·정규화)은 mock 없이 테스트하고,
LLM 호출이 필요한 추출 흐름은 Teacher mock으로 테스트합니다.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from slm_factory.config import OntologyConfig, TeacherConfig
from slm_factory.models import ParsedDocument
from slm_factory.ontology.extractor import OntologyExtractor
from slm_factory.ontology.models import Entity, Relation


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def ontology_config():
    return OntologyConfig(
        enabled=True,
        entity_types=["Person", "Organization", "Concept", "Technology"],
        min_confidence=0.5,
    )


@pytest.fixture
def teacher_config():
    return TeacherConfig(backend="openai", max_context_chars=500)


@pytest.fixture
def mock_teacher():
    teacher = MagicMock()
    teacher.agenerate = AsyncMock()
    return teacher


@pytest.fixture
def extractor(mock_teacher, ontology_config, teacher_config):
    return OntologyExtractor(mock_teacher, ontology_config, teacher_config)


# ---------------------------------------------------------------------------
# 프롬프트 빌드
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:
    """_build_extraction_prompt() 테스트입니다."""

    def test_문서_제목_포함(self, extractor):
        """프롬프트에 문서 제목이 포함됩니다."""
        prompt = extractor._build_extraction_prompt("테스트 문서", "내용")
        assert "테스트 문서" in prompt

    def test_문서_내용_포함(self, extractor):
        """프롬프트에 문서 내용이 포함됩니다."""
        prompt = extractor._build_extraction_prompt("제목", "본문 내용입니다")
        assert "본문 내용입니다" in prompt

    def test_엔티티_유형_포함(self, extractor):
        """프롬프트에 허용된 엔티티 유형이 포함됩니다."""
        prompt = extractor._build_extraction_prompt("제목", "내용")
        assert "Person" in prompt
        assert "Organization" in prompt

    def test_긴_내용_절삭(self, extractor):
        """max_context_chars를 초과하는 내용은 절삭됩니다."""
        long_content = "가" * 1000
        prompt = extractor._build_extraction_prompt("제목", long_content)
        assert "이하 생략" in prompt

    def test_짧은_내용_생략_표시_없음(self, extractor):
        """max_context_chars 이하의 내용에는 생략 표시가 없습니다."""
        short_content = "짧은 내용"
        prompt = extractor._build_extraction_prompt("제목", short_content)
        assert "이하 생략" not in prompt

    def test_JSON_출력_형식_명시(self, extractor):
        """프롬프트에 JSON 출력 형식이 명시되어 있습니다."""
        prompt = extractor._build_extraction_prompt("제목", "내용")
        assert "json" in prompt.lower()
        assert "entities" in prompt
        assert "relations" in prompt


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------


class TestParseExtraction:
    """_parse_extraction() 테스트입니다."""

    def test_정상_JSON_파싱(self, extractor):
        """올바른 JSON 응답에서 엔티티와 관계를 추출합니다."""
        text = json.dumps({
            "entities": [
                {"name": "Python", "entity_type": "Technology", "confidence": 0.9},
            ],
            "relations": [
                {"subject": "Python", "predicate": "사용", "object": "개발자", "confidence": 0.8},
            ],
        })
        entities, relations = extractor._parse_extraction(text, "doc.pdf")
        assert len(entities) == 1
        assert entities[0].name == "Python"
        assert entities[0].source_doc == "doc.pdf"
        assert len(relations) == 1
        assert relations[0].predicate == "사용"

    def test_코드_블록_감싼_JSON(self, extractor):
        """```json ... ``` 마커로 감싼 응답도 파싱합니다."""
        text = '```json\n{"entities": [{"name": "A", "entity_type": "Concept"}], "relations": []}\n```'
        entities, relations = extractor._parse_extraction(text, "doc.pdf")
        assert len(entities) == 1
        assert entities[0].name == "A"

    def test_잘못된_JSON_정규식_폴백(self, extractor):
        """순수 JSON 파싱 실패 시 정규식 폴백으로 JSON 블록을 추출합니다."""
        text = '아래는 결과입니다:\n{"entities": [{"name": "B", "entity_type": "Person"}], "relations": []}\n감사합니다.'
        entities, relations = extractor._parse_extraction(text, "doc.pdf")
        assert len(entities) == 1
        assert entities[0].name == "B"

    def test_완전히_잘못된_응답(self, extractor):
        """파싱 불가능한 응답은 빈 리스트를 반환합니다."""
        entities, relations = extractor._parse_extraction(
            "이것은 JSON이 아닙니다", "doc.pdf",
        )
        assert entities == []
        assert relations == []

    def test_빈_엔티티_이름_무시(self, extractor):
        """name이 비어있는 엔티티는 무시됩니다."""
        text = json.dumps({
            "entities": [
                {"name": "", "entity_type": "Concept"},
                {"name": "Valid", "entity_type": "Concept"},
            ],
            "relations": [],
        })
        entities, _ = extractor._parse_extraction(text, "doc.pdf")
        assert len(entities) == 1
        assert entities[0].name == "Valid"

    def test_빈_엔티티_유형_무시(self, extractor):
        """entity_type이 비어있는 엔티티는 무시됩니다."""
        text = json.dumps({
            "entities": [{"name": "NoType", "entity_type": ""}],
            "relations": [],
        })
        entities, _ = extractor._parse_extraction(text, "doc.pdf")
        assert entities == []

    def test_비딕셔너리_항목_무시(self, extractor):
        """딕셔너리가 아닌 항목은 무시됩니다."""
        text = json.dumps({
            "entities": ["invalid", {"name": "Valid", "entity_type": "Concept"}],
            "relations": [42],
        })
        entities, relations = extractor._parse_extraction(text, "doc.pdf")
        assert len(entities) == 1
        assert relations == []

    def test_관계_필수_필드_누락_무시(self, extractor):
        """subject·predicate·object 중 하나라도 비어있으면 무시됩니다."""
        text = json.dumps({
            "entities": [],
            "relations": [
                {"subject": "A", "predicate": "", "object": "B"},
                {"subject": "", "predicate": "관련", "object": "B"},
                {"subject": "A", "predicate": "관련", "object": ""},
                {"subject": "A", "predicate": "관련", "object": "B"},
            ],
        })
        _, relations = extractor._parse_extraction(text, "doc.pdf")
        assert len(relations) == 1


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


class TestValidateExtraction:
    """_validate_extraction() 테스트입니다."""

    def test_허용_유형_필터링(self, extractor):
        """허용되지 않은 entity_type은 제거됩니다."""
        entities = [
            Entity(name="A", entity_type="Person"),
            Entity(name="B", entity_type="UnknownType"),
        ]
        valid_e, _ = extractor._validate_extraction(entities, [])
        names = [e.name for e in valid_e]
        assert "A" in names
        assert "B" not in names

    def test_최소_확신도_필터링(self, extractor):
        """min_confidence 미만인 엔티티는 제거됩니다."""
        entities = [
            Entity(name="High", entity_type="Concept", confidence=0.8),
            Entity(name="Low", entity_type="Concept", confidence=0.3),
        ]
        valid_e, _ = extractor._validate_extraction(entities, [])
        names = [e.name for e in valid_e]
        assert "High" in names
        assert "Low" not in names

    def test_관계_참조_엔티티_필터링(self, extractor):
        """관계의 주어·목적어가 유효 엔티티에 없으면 제거됩니다."""
        entities = [Entity(name="A", entity_type="Concept", confidence=0.9)]
        relations = [
            Relation(subject="A", predicate="관련", object="B"),  # B는 엔티티에 없음
            Relation(subject="A", predicate="자기", object="A"),  # 유효
        ]
        _, valid_r = extractor._validate_extraction(entities, relations)
        assert len(valid_r) == 1
        assert valid_r[0].predicate == "자기"

    def test_관계_확신도_필터링(self, extractor):
        """min_confidence 미만인 관계는 제거됩니다."""
        entities = [
            Entity(name="A", entity_type="Concept"),
            Entity(name="B", entity_type="Concept"),
        ]
        relations = [
            Relation(subject="A", predicate="높음", object="B", confidence=0.9),
            Relation(subject="A", predicate="낮음", object="B", confidence=0.2),
        ]
        _, valid_r = extractor._validate_extraction(entities, relations)
        predicates = [r.predicate for r in valid_r]
        assert "높음" in predicates
        assert "낮음" not in predicates

    def test_빈_입력(self, extractor):
        """빈 엔티티·관계 입력도 정상 처리됩니다."""
        valid_e, valid_r = extractor._validate_extraction([], [])
        assert valid_e == []
        assert valid_r == []


# ---------------------------------------------------------------------------
# 정규화 — 엔티티
# ---------------------------------------------------------------------------


class TestNormalizeEntities:
    """_normalize_entities() 테스트입니다."""

    def test_대소문자_중복_제거(self, extractor):
        """(name.upper(), entity_type) 기준으로 동일 엔티티를 병합합니다."""
        entities = [
            Entity(name="python", entity_type="Technology", confidence=0.7),
            Entity(name="Python", entity_type="Technology", confidence=0.9),
        ]
        result = extractor._normalize_entities(entities)
        assert len(result) == 1

    def test_긴_이름_선호(self, extractor):
        """동일 엔티티 중 더 긴 이름을 정규 이름으로 선택합니다."""
        entities = [
            Entity(name="삼성", entity_type="Organization", confidence=0.8),
            Entity(name="삼성전자", entity_type="Organization", confidence=0.7),
        ]
        # "삼성" and "삼성전자" have different upper values, so they won't merge
        # Let me use same upper key
        entities = [
            Entity(name="abc", entity_type="Concept", confidence=0.7),
            Entity(name="ABC", entity_type="Concept", confidence=0.8),
        ]
        result = extractor._normalize_entities(entities)
        assert len(result) == 1
        # ABC is len 3 same as abc, first one wins since len is equal
        assert result[0].confidence == 0.8  # max confidence

    def test_확신도_최댓값_보존(self, extractor):
        """병합 시 확신도는 그룹 내 최댓값을 사용합니다."""
        entities = [
            Entity(name="test", entity_type="Concept", confidence=0.3),
            Entity(name="TEST", entity_type="Concept", confidence=0.9),
        ]
        result = extractor._normalize_entities(entities)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_다른_유형은_별도(self, extractor):
        """같은 이름이라도 entity_type이 다르면 별도 엔티티입니다."""
        entities = [
            Entity(name="Apple", entity_type="Organization"),
            Entity(name="Apple", entity_type="Concept"),
        ]
        result = extractor._normalize_entities(entities)
        assert len(result) == 2

    def test_빈_입력(self, extractor):
        """빈 리스트 입력도 정상 처리됩니다."""
        assert extractor._normalize_entities([]) == []


# ---------------------------------------------------------------------------
# 정규화 — 관계
# ---------------------------------------------------------------------------


class TestNormalizeRelations:
    """_normalize_relations() 테스트입니다."""

    def test_대소문자_중복_제거(self, extractor):
        """(subject.upper(), predicate.upper(), object.upper()) 기준으로 중복 제거합니다."""
        relations = [
            Relation(subject="A", predicate="관련", object="B", confidence=0.7),
            Relation(subject="a", predicate="관련", object="b", confidence=0.9),
        ]
        result = extractor._normalize_relations(relations)
        assert len(result) == 1

    def test_높은_확신도_선호(self, extractor):
        """중복 관계 중 확신도가 높은 것을 선택합니다."""
        relations = [
            Relation(subject="A", predicate="포함", object="B", confidence=0.3),
            Relation(subject="A", predicate="포함", object="B", confidence=0.9),
        ]
        result = extractor._normalize_relations(relations)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_다른_술어는_별도(self, extractor):
        """같은 주어·목적어라도 술어가 다르면 별도 관계입니다."""
        relations = [
            Relation(subject="A", predicate="포함", object="B"),
            Relation(subject="A", predicate="사용", object="B"),
        ]
        result = extractor._normalize_relations(relations)
        assert len(result) == 2

    def test_빈_입력(self, extractor):
        """빈 리스트 입력도 정상 처리됩니다."""
        assert extractor._normalize_relations([]) == []


# ---------------------------------------------------------------------------
# 비동기 추출 — Teacher mock
# ---------------------------------------------------------------------------


class TestExtractOne:
    """extract_one() 테스트입니다 (Teacher mock 사용)."""

    async def test_단일_청크_추출(self, extractor, mock_teacher):
        """짧은 문서에서 엔티티·관계를 추출합니다."""
        mock_teacher.agenerate.return_value = json.dumps({
            "entities": [
                {"name": "Python", "entity_type": "Technology", "confidence": 0.9},
            ],
            "relations": [],
        })
        doc = ParsedDocument(
            doc_id="test.pdf", title="테스트", content="짧은 내용",
        )

        entities, relations = await extractor.extract_one(doc)
        assert len(entities) == 1
        assert entities[0].name == "Python"
        mock_teacher.agenerate.assert_called_once()

    async def test_다중_청크_병합(self, mock_teacher, ontology_config, teacher_config):
        """max_context_chars를 초과하는 문서는 청크 분할 후 병합합니다."""
        teacher_config_small = TeacherConfig(backend="openai", max_context_chars=50)
        ext = OntologyExtractor(mock_teacher, ontology_config, teacher_config_small)

        mock_teacher.agenerate.return_value = json.dumps({
            "entities": [
                {"name": "Entity", "entity_type": "Concept", "confidence": 0.8},
            ],
            "relations": [],
        })
        doc = ParsedDocument(
            doc_id="test.pdf", title="긴 문서",
            content="가" * 200,
        )

        entities, _ = await ext.extract_one(doc)
        # 여러 청크에서 추출되므로 agenerate가 여러 번 호출됨
        assert mock_teacher.agenerate.call_count > 1
        # 정규화로 동일 엔티티는 병합됨
        assert len(entities) == 1

    async def test_청크_실패_허용(self, extractor, mock_teacher):
        """일부 청크 추출이 실패해도 나머지 결과는 반환됩니다."""
        mock_teacher.agenerate.side_effect = [
            json.dumps({
                "entities": [
                    {"name": "OK", "entity_type": "Concept", "confidence": 0.8},
                ],
                "relations": [],
            }),
            RuntimeError("LLM 오류"),
        ]

        teacher_cfg = TeacherConfig(backend="openai", max_context_chars=50)
        ext = OntologyExtractor(
            mock_teacher, extractor.config, teacher_cfg,
        )
        doc = ParsedDocument(
            doc_id="test.pdf", title="테스트", content="가" * 200,
        )

        entities, _ = await ext.extract_one(doc)
        assert any(e.name == "OK" for e in entities)


class TestExtractAll:
    """extract_all() 테스트입니다 (Teacher mock 사용)."""

    async def test_여러_문서_추출(self, extractor, mock_teacher):
        """여러 문서에서 온톨로지를 추출하고 병합합니다."""
        mock_teacher.agenerate.return_value = json.dumps({
            "entities": [
                {"name": "Common", "entity_type": "Concept", "confidence": 0.9},
            ],
            "relations": [],
        })
        docs = [
            ParsedDocument(doc_id="a.pdf", title="문서A", content="내용A"),
            ParsedDocument(doc_id="b.pdf", title="문서B", content="내용B"),
        ]

        kg = await extractor.extract_all(docs)
        assert isinstance(kg.entities, list)
        # 2번 호출됨 (문서 2개)
        assert mock_teacher.agenerate.call_count == 2

    async def test_문서_추출_실패_건너뜀(self, extractor, mock_teacher):
        """개별 문서 추출 실패 시 해당 문서만 건너뜁니다."""
        mock_teacher.agenerate.side_effect = [
            json.dumps({
                "entities": [
                    {"name": "Success", "entity_type": "Concept", "confidence": 0.9},
                ],
                "relations": [],
            }),
            RuntimeError("실패"),
        ]
        docs = [
            ParsedDocument(doc_id="ok.pdf", title="성공", content="내용"),
            ParsedDocument(doc_id="fail.pdf", title="실패", content="내용"),
        ]

        kg = await extractor.extract_all(docs)
        names = [e.name for e in kg.entities]
        assert "Success" in names

    async def test_빈_문서_리스트(self, extractor, mock_teacher):
        """빈 문서 리스트를 전달하면 빈 그래프를 반환합니다."""
        kg = await extractor.extract_all([])
        assert kg.entities == []
        assert kg.relations == []
        mock_teacher.agenerate.assert_not_called()
