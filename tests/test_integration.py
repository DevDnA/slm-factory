"""청킹·온톨로지·재생성 기능의 통합 테스트입니다.

개별 모듈 단위가 아닌 모듈 간 데이터 흐름을 검증합니다.
LLM 호출은 mock하지만 데이터 구조는 실제 객체를 사용합니다.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slm_factory.config import SLMConfig
from slm_factory.models import ParsedDocument, QAPair
from slm_factory.ontology.models import Entity, KnowledgeGraph, Relation
from slm_factory.pipeline import Pipeline
from slm_factory.teacher.qa_generator import QAGenerator, chunk_document


def _make_pipeline(make_config, tmp_path, **overrides) -> Pipeline:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = {"paths": {"output": str(output_dir), "documents": str(docs_dir)}}
    defaults.update(overrides)
    config = make_config(**defaults)
    return Pipeline(config)


def _long_content(paragraphs: int = 10, chars_per_paragraph: int = 2000) -> str:
    return "\n\n".join(
        f"문단 {i + 1}. " + "한국어 테스트 문장입니다. " * (chars_per_paragraph // 15)
        for i in range(paragraphs)
    )


# ---------------------------------------------------------------------------
# 청킹 → QA 생성 체인
# ---------------------------------------------------------------------------


class TestChunkingToQAChain:
    """청킹이 활성화되면 QA 생성이 청크 단위로 동작하는지 검증합니다."""

    def test_청킹_활성화시_여러_청크에서_QA_생성(self, make_config, mocker):
        """긴 문서에서 청킹이 활성화되면 chunk별로 QA 생성 호출이 발생하는지 확인합니다."""
        config = make_config(
            chunking={"enabled": True, "chunk_size": 3000, "overlap_chars": 200},
            teacher={"max_context_chars": 12000},
            questions={"categories": {"test": ["테스트 질문?"]}},
        )

        long_content = _long_content(paragraphs=8, chars_per_paragraph=1500)
        doc = ParsedDocument(
            doc_id="long.pdf", title="긴 문서", content=long_content,
            tables=[], metadata={},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.questions_config = config.questions
        generator.teacher_config = config.teacher
        generator.chunking_config = config.chunking
        generator.max_context = config.teacher.max_context_chars

        chunks = generator._get_doc_chunks(doc)

        assert len(chunks) > 1
        for chunk_content, chunk_info in chunks:
            assert len(chunk_content) > 0
        assert chunks[-1][1] is not None
        assert "Part" in chunks[0][1]

    def test_청킹_비활성화시_단일_청크_반환(self, make_config):
        """청킹 비활성화 시 원본 문서가 단일 청크로 반환되는지 확인합니다."""
        config = make_config(
            chunking={"enabled": False},
            teacher={"max_context_chars": 12000},
        )

        doc = ParsedDocument(
            doc_id="test.pdf", title="문서", content="짧은 내용",
            tables=[], metadata={},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.chunking_config = config.chunking
        generator.max_context = config.teacher.max_context_chars

        chunks = generator._get_doc_chunks(doc)
        assert len(chunks) == 1
        assert chunks[0] == ("짧은 내용", None)

    def test_청크_정보가_프롬프트에_포함(self, make_config):
        """chunk_info가 프롬프트의 문서 제목에 반영되는지 확인합니다."""
        config = make_config(
            teacher={"max_context_chars": 50000},
            questions={"categories": {"test": ["질문?"]}},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.questions_config = config.questions
        generator.teacher_config = config.teacher
        generator.max_context = config.teacher.max_context_chars

        prompt = generator.build_prompt(
            doc_title="테스트 문서",
            content="내용",
            question="질문?",
            chunk_info="Part 2/5",
        )

        assert "테스트 문서 (Part 2/5)" in prompt

    def test_파이프라인_step_generate에서_청킹_동작(self, make_config, tmp_path, mocker):
        """Pipeline.step_generate가 청킹 활성화 시 generate_all_async를 올바르게 호출하는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            chunking={"enabled": True, "chunk_size": 5000, "overlap_chars": 300},
            questions={"categories": {"test": ["질문?"]}},
        )

        doc = ParsedDocument(
            doc_id="test.pdf", title="문서", content=_long_content(5, 2000),
            tables=[], metadata={},
        )

        mock_qa = QAPair(
            question="질문?", answer="답변", instruction="질문?",
            source_doc="test.pdf", category="",
        )

        mock_generator_cls = mocker.patch("slm_factory.teacher.qa_generator.QAGenerator")
        mock_generator = mock_generator_cls.return_value
        mock_generator.save_alpaca = MagicMock()

        mock_coro = AsyncMock(return_value=[mock_qa])
        mock_generator.generate_all_async = mock_coro
        mocker.patch("slm_factory.pipeline.run_async", side_effect=lambda c: asyncio.run(c) if asyncio.iscoroutine(c) else c)
        mocker.patch("slm_factory.pipeline.run_async", return_value=[mock_qa])

        pairs = pipeline.step_generate([doc])

        assert len(pairs) == 1
        mock_generator.generate_all_async.assert_called_once()


# ---------------------------------------------------------------------------
# 온톨로지 멀티청크 → QA 연동
# ---------------------------------------------------------------------------


class TestOntologyMultiChunkToQA:
    """온톨로지가 멀티청크로 추출되어 QA 생성에 컨텍스트로 주입되는지 검증합니다."""

    def test_온톨로지_멀티청크_추출_결과_병합(self, make_config, mocker):
        """긴 문서에서 멀티청크 온톨로지 추출 시 엔티티가 병합되는지 확인합니다."""
        from slm_factory.ontology.extractor import OntologyExtractor

        config = make_config(
            ontology={"enabled": True, "entity_types": ["Person", "Organization"]},
            teacher={"max_context_chars": 3000},
        )

        mock_teacher = AsyncMock()
        mock_teacher.agenerate = AsyncMock(return_value=json.dumps({
            "entities": [
                {"name": "홍길동", "entity_type": "Person", "confidence": 0.9},
                {"name": "모비젠", "entity_type": "Organization", "confidence": 0.8},
            ],
            "relations": [
                {"subject": "홍길동", "predicate": "소속", "object": "모비젠", "confidence": 0.85},
            ],
        }))

        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)

        long_content = _long_content(6, 2000)
        doc = ParsedDocument(
            doc_id="test.pdf", title="테스트", content=long_content,
            tables=[], metadata={},
        )

        entities, relations = asyncio.run(extractor.extract_one(doc))

        assert mock_teacher.agenerate.call_count > 1
        assert len(entities) >= 1
        assert any(e.name == "홍길동" for e in entities)

    def test_온톨로지_컨텍스트가_QA_프롬프트에_주입(self, make_config):
        """ontology.enrich_qa=True일 때 온톨로지 컨텍스트가 프롬프트에 포함되는지 확인합니다."""
        config = make_config(
            ontology={"enabled": True, "enrich_qa": True},
            teacher={"max_context_chars": 50000},
            questions={"categories": {"test": ["질문?"]}},
        )

        generator = QAGenerator.__new__(QAGenerator)
        generator.config = config
        generator.questions_config = config.questions
        generator.teacher_config = config.teacher
        generator.max_context = config.teacher.max_context_chars

        ontology_ctx = "엔티티: 홍길동 (Person), 모비젠 (Organization)\n관계: 홍길동 → 소속 → 모비젠"

        prompt = generator.build_prompt(
            doc_title="문서",
            content="내용",
            question="질문?",
            ontology_context=ontology_ctx,
        )

        assert "홍길동" in prompt
        assert "모비젠" in prompt
        assert "관련 지식" in prompt

    def test_파이프라인에서_온톨로지가_QA생성에_전달(self, make_config, tmp_path, mocker):
        """Pipeline.step_generate가 ontology를 generate_all_async의 ontology_context로 변환하는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            ontology={"enabled": True, "enrich_qa": True},
        )

        doc = ParsedDocument(
            doc_id="test.pdf", title="문서제목", content="내용",
            tables=[], metadata={},
        )

        kg = KnowledgeGraph(
            entities=[Entity(name="테스트", entity_type="Concept", source_doc="문서제목", confidence=0.9)],
            relations=[],
        )

        mock_qa = QAPair(
            question="Q", answer="A", instruction="Q",
            source_doc="test.pdf", category="",
        )

        mocker.patch("slm_factory.pipeline.run_async", return_value=[mock_qa])
        mock_generator_cls = mocker.patch("slm_factory.teacher.qa_generator.QAGenerator")
        mock_generator = mock_generator_cls.return_value
        mock_generator.save_alpaca = MagicMock()

        pairs = pipeline.step_generate([doc], ontology=kg)

        call_kwargs = mocker.patch("slm_factory.pipeline.run_async").call_args
        assert pairs == [mock_qa]


# ---------------------------------------------------------------------------
# 스코어 → 재생성 루프
# ---------------------------------------------------------------------------


class TestScoreRegenerationLoop:
    """점수 평가 후 저품질 QA를 재생성하는 루프를 검증합니다."""

    def test_재생성이_낮은_점수_QA를_복구(self, make_config, make_qa_pair, make_parsed_doc, tmp_path, mocker):
        """재생성 활성화 시 낮은 점수 QA가 재생성되어 복구되는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            scoring={"enabled": True, "threshold": 3.0, "regenerate": True, "max_regenerate_rounds": 2},
        )

        good_pair = make_qa_pair(question="좋은 질문", answer="충분히 긴 좋은 답변입니다. 정확하고 상세한 내용을 포함합니다.")
        bad_pair = make_qa_pair(question="나쁜 질문", answer="충분히 긴 나쁜 답변입니다. 부정확한 내용이 포함되어 있습니다.")
        doc = make_parsed_doc(doc_id="test.pdf")

        mock_teacher = MagicMock()
        regen_response = json.dumps({
            "instruction": "나쁜 질문",
            "output": "충분히 긴 재생성된 개선 답변입니다. 정확하고 상세한 내용을 포함합니다.",
        })

        async def mock_agenerate(prompt, **kwargs):
            return regen_response

        mock_teacher.agenerate = mock_agenerate

        mocker.patch("slm_factory.teacher.create_teacher", return_value=mock_teacher)

        call_count = [0]

        async def mock_score_all(pairs, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [good_pair], [(bad_pair, 2, "부정확")]
            else:
                return list(pairs), []

        mock_scorer_cls = mocker.patch("slm_factory.scorer.QualityScorer")
        mock_scorer = mock_scorer_cls.return_value
        mock_scorer.score_all = mock_score_all

        mocker.patch("slm_factory.pipeline.run_async", side_effect=lambda c: asyncio.run(c))

        result = pipeline.step_score([good_pair, bad_pair], docs=[doc])

        assert len(result) >= 2

    def test_재생성_프롬프트에_이전_점수_피드백_포함(self, make_config, make_qa_pair, make_parsed_doc, tmp_path, mocker):
        """재생성 프롬프트에 이전 점수와 이유가 포함되는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            scoring={"enabled": True, "threshold": 3.0, "regenerate": True, "max_regenerate_rounds": 1},
        )

        bad_pair = make_qa_pair(question="질문", source_doc="test.pdf")
        doc = make_parsed_doc(doc_id="test.pdf")

        captured_prompts = []

        mock_teacher = MagicMock()

        async def capture_agenerate(prompt, **kwargs):
            captured_prompts.append(prompt)
            return json.dumps({"instruction": "질문", "output": "개선된 답변입니다. 충분히 길고 정확합니다."})

        mock_teacher.agenerate = capture_agenerate

        mocker.patch("slm_factory.teacher.create_teacher", return_value=mock_teacher)

        async def mock_score_all(pairs, **kwargs):
            return [], [(p, 2, "불완전한 답변") for p in pairs]

        mock_scorer_cls = mocker.patch("slm_factory.scorer.QualityScorer")
        mock_scorer_cls.return_value.score_all = mock_score_all

        mocker.patch("slm_factory.pipeline.run_async", side_effect=lambda c: asyncio.run(c))

        pipeline.step_score([bad_pair], docs=[doc])

        assert len(captured_prompts) > 0
        assert "2/5점" in captured_prompts[0]
        assert "불완전한 답변" in captured_prompts[0]

    def test_docs_없으면_재생성_건너뜀(self, make_config, make_qa_pair, tmp_path, mocker):
        """docs가 None이면 재생성을 건너뛰는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            scoring={"enabled": True, "threshold": 3.0, "regenerate": True},
        )

        pair = make_qa_pair()

        mocker.patch("slm_factory.teacher.create_teacher", return_value=MagicMock())
        mock_scorer_cls = mocker.patch("slm_factory.scorer.QualityScorer")

        mocker.patch(
            "slm_factory.pipeline.run_async",
            return_value=([pair], [(pair, 1, "나쁨")]),
        )

        result = pipeline.step_score([pair, pair])

        assert len(result) >= 1


# ---------------------------------------------------------------------------
# 전체 체인 통합 + 엣지 케이스
# ---------------------------------------------------------------------------


class TestFullChainIntegration:
    """전체 파이프라인 체인(파싱→청킹→온톨로지→QA→검증→스코어→재생성)을 검증합니다."""

    def test_run_메서드에서_전체_체인_실행(self, make_config, tmp_path, mocker):
        """Pipeline.run()이 청킹+온톨로지+재생성을 포함해 전체 체인을 실행하는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            chunking={"enabled": True, "chunk_size": 5000, "overlap_chars": 300},
            ontology={"enabled": True, "enrich_qa": True},
            scoring={"enabled": True, "threshold": 3.0, "regenerate": True},
        )

        mock_docs = [MagicMock()]
        mock_pairs = [MagicMock()]
        mock_kg = KnowledgeGraph()
        mock_training_path = tmp_path / "training.jsonl"
        mock_adapter = tmp_path / "adapter"
        mock_export = tmp_path / "export"

        mocker.patch.object(pipeline, "step_parse", return_value=mock_docs)
        mocker.patch.object(pipeline, "step_extract_ontology", return_value=mock_kg)
        mocker.patch.object(pipeline, "step_generate", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_validate", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_score", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_augment", return_value=mock_pairs)
        mocker.patch.object(pipeline, "step_analyze")
        mocker.patch.object(pipeline, "step_convert", return_value=mock_training_path)
        mocker.patch.object(pipeline, "step_train", return_value=mock_adapter)
        mocker.patch.object(pipeline, "step_export", return_value=mock_export)

        pipeline.run()

        pipeline.step_generate.assert_called_once_with(mock_docs, ontology=mock_kg)
        pipeline.step_score.assert_called_once_with(mock_pairs, docs=mock_docs, ontology=mock_kg)


class TestEdgeCases:
    """새 기능의 엣지 케이스를 검증합니다."""

    def test_빈_문서_청킹시_단일_빈_청크(self):
        """빈 문서를 청킹하면 빈 문자열 하나가 반환되는지 확인합니다."""
        result = chunk_document("", chunk_size=5000, overlap=500)
        assert result == [""]

    def test_청크사이즈보다_짧은_문서(self):
        """chunk_size보다 짧은 문서는 분할 없이 반환되는지 확인합니다."""
        short = "짧은 문서입니다."
        result = chunk_document(short, chunk_size=10000, overlap=500)
        assert result == [short]

    def test_문단_경계_없는_긴_문서_강제_분할(self):
        """\\n\\n 없는 긴 문자열도 chunk_size에서 강제 분할되는지 확인합니다."""
        no_paragraphs = "가" * 25000
        result = chunk_document(no_paragraphs, chunk_size=10000, overlap=500)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= 10000

    def test_단일_문단_긴_문서(self):
        """단일 문단이지만 chunk_size를 초과하는 문서가 분할되는지 확인합니다."""
        single_paragraph = "이것은 매우 긴 단일 문단입니다. " * 1000
        result = chunk_document(single_paragraph, chunk_size=5000, overlap=300)
        assert len(result) > 1

    def test_온톨로지_extract_one_한_청크_실패시_나머지_유지(self, make_config, mocker):
        """멀티청크 온톨로지 추출 중 한 청크가 실패해도 나머지 결과를 유지하는지 확인합니다."""
        from slm_factory.ontology.extractor import OntologyExtractor

        config = make_config(
            ontology={"enabled": True, "entity_types": ["Person"]},
            teacher={"max_context_chars": 3000},
        )

        call_count = [0]

        async def flaky_agenerate(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM 연결 실패")
            return json.dumps({
                "entities": [{"name": "테스트", "entity_type": "Person", "confidence": 0.9}],
                "relations": [],
            })

        mock_teacher = AsyncMock()
        mock_teacher.agenerate = flaky_agenerate

        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)

        doc = ParsedDocument(
            doc_id="test.pdf", title="문서",
            content=_long_content(4, 2000),
            tables=[], metadata={},
        )

        entities, relations = asyncio.run(extractor.extract_one(doc))

        assert call_count[0] > 1
        assert len(entities) >= 1

    def test_재생성_max_rounds_초과시_중단(self, make_config, make_qa_pair, make_parsed_doc, tmp_path, mocker):
        """max_regenerate_rounds를 초과하면 재생성이 중단되는지 확인합니다."""
        pipeline = _make_pipeline(
            make_config, tmp_path,
            scoring={"enabled": True, "threshold": 3.0, "regenerate": True, "max_regenerate_rounds": 1},
        )

        bad_pair = make_qa_pair(source_doc="test.pdf")
        doc = make_parsed_doc(doc_id="test.pdf")

        mock_teacher = MagicMock()
        regen_response = json.dumps({
            "instruction": "질문", "output": "여전히 부족한 답변입니다. 개선이 필요합니다.",
        })

        async def mock_agenerate(prompt, **kwargs):
            return regen_response

        mock_teacher.agenerate = mock_agenerate

        mocker.patch("slm_factory.teacher.create_teacher", return_value=mock_teacher)

        async def always_fail_score(pairs, **kwargs):
            return [], [(p, 1, "여전히 나쁨") for p in pairs]

        mock_scorer_cls = mocker.patch("slm_factory.scorer.QualityScorer")
        mock_scorer_cls.return_value.score_all = always_fail_score

        mocker.patch("slm_factory.pipeline.run_async", side_effect=lambda c: asyncio.run(c))

        result = pipeline.step_score([bad_pair], docs=[doc])

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Relation 중복 제거 테스트
# ---------------------------------------------------------------------------


class TestRelationNormalization:
    """온톨로지 멀티청크 추출 시 Relation 중복 제거 검증."""

    def test_동일_관계_중복_제거(self, make_config):
        """동일 subject-predicate-object 관계가 하나로 합쳐집니다."""
        from slm_factory.ontology.extractor import OntologyExtractor
        from slm_factory.ontology.models import Relation

        config = make_config()
        mock_teacher = MagicMock()
        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)

        relations = [
            Relation(subject="삼성전자", predicate="개발", object="갤럭시", source_doc="d1", confidence=0.8),
            Relation(subject="삼성전자", predicate="개발", object="갤럭시", source_doc="d1", confidence=0.9),
            Relation(subject="삼성전자", predicate="개발", object="갤럭시", source_doc="d1", confidence=0.7),
        ]

        result = extractor._normalize_relations(relations)

        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_대소문자_무시_중복_제거(self, make_config):
        """대소문자가 달라도 같은 관계로 인식합니다."""
        from slm_factory.ontology.extractor import OntologyExtractor
        from slm_factory.ontology.models import Relation

        config = make_config()
        mock_teacher = MagicMock()
        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)

        relations = [
            Relation(subject="Samsung", predicate="develops", object="Galaxy", confidence=0.8),
            Relation(subject="SAMSUNG", predicate="DEVELOPS", object="GALAXY", confidence=0.6),
        ]

        result = extractor._normalize_relations(relations)

        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_다른_관계_유지(self, make_config):
        """서로 다른 관계는 모두 유지됩니다."""
        from slm_factory.ontology.extractor import OntologyExtractor
        from slm_factory.ontology.models import Relation

        config = make_config()
        mock_teacher = MagicMock()
        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)

        relations = [
            Relation(subject="A", predicate="소속", object="B"),
            Relation(subject="A", predicate="개발", object="C"),
            Relation(subject="B", predicate="소속", object="C"),
        ]

        result = extractor._normalize_relations(relations)

        assert len(result) == 3

    def test_빈_관계_리스트(self, make_config):
        """빈 리스트 입력 시 빈 리스트를 반환합니다."""
        from slm_factory.ontology.extractor import OntologyExtractor

        config = make_config()
        mock_teacher = MagicMock()
        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)

        result = extractor._normalize_relations([])

        assert result == []

    def test_extract_one에서_관계_중복_제거_적용(self, make_config):
        """extract_one이 멀티청크 추출 후 관계를 정규화합니다."""
        from slm_factory.ontology.extractor import OntologyExtractor
        from slm_factory.ontology.models import Entity, Relation
        from slm_factory.models import ParsedDocument

        config = make_config(ontology={"enabled": True})
        mock_teacher = MagicMock()

        chunk1_response = json.dumps({"entities": [
            {"name": "X", "entity_type": "Concept"},
            {"name": "Y", "entity_type": "Concept"},
        ], "relations": [
            {"subject": "X", "predicate": "관련", "object": "Y"}
        ]})
        chunk2_response = json.dumps({"entities": [
            {"name": "X", "entity_type": "Concept"},
            {"name": "Y", "entity_type": "Concept"},
        ], "relations": [
            {"subject": "X", "predicate": "관련", "object": "Y"}
        ]})

        call_count = 0

        async def mock_agenerate(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return chunk1_response if call_count == 1 else chunk2_response

        mock_teacher.agenerate = mock_agenerate

        extractor = OntologyExtractor(mock_teacher, config.ontology, config.teacher)
        doc = ParsedDocument(
            doc_id="test", title="테스트", content="A" * 20000,
        )

        entities, relations = asyncio.run(extractor.extract_one(doc))

        assert len(relations) == 1
