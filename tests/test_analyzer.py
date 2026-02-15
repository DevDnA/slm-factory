"""DataAnalyzer 및 AnalysisReport 테스트 — fixture 기반, mock 불필요."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_factory.analyzer import AnalysisReport, DataAnalyzer
from slm_factory.models import QAPair


@pytest.fixture
def analyzer():
    return DataAnalyzer()


@pytest.fixture
def sample_pairs():
    return [
        QAPair(question="질문1?", answer="답변1입니다.", source_doc="doc1.pdf", category="general"),
        QAPair(question="질문2?", answer="답변2입니다. 좀 더 긴 답변.", source_doc="doc1.pdf", category="general"),
        QAPair(question="질문3?", answer="답변3.", source_doc="doc2.pdf", category="technical"),
    ]


@pytest.fixture
def augmented_pairs():
    return [
        QAPair(question="원본 질문?", answer="답변입니다.", source_doc="doc1.pdf", category="general", is_augmented=False),
        QAPair(question="패러프레이즈 질문?", answer="답변입니다.", source_doc="doc1.pdf", category="general", is_augmented=True),
        QAPair(question="또 다른 패러프레이즈?", answer="답변입니다.", source_doc="doc1.pdf", category="general", is_augmented=True),
    ]


class TestAnalysisReport:
    def test_default_values(self):
        report = AnalysisReport()
        assert report.total_pairs == 0
        assert report.original_pairs == 0
        assert report.augmented_pairs == 0
        assert report.category_distribution == {}
        assert report.source_doc_distribution == {}
        assert report.answer_length_stats == {}
        assert report.question_length_stats == {}
        assert report.quality_score_stats == {}
        assert report.warnings == []


class TestDataAnalyzer:
    def test_analyze_empty_list(self, analyzer):
        report = analyzer.analyze([])
        assert report.total_pairs == 0
        assert "데이터가 비어 있습니다." in report.warnings

    def test_analyze_normal_data(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        assert report.total_pairs == 3
        assert report.original_pairs == 3
        assert report.augmented_pairs == 0

    def test_category_distribution(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        assert "general" in report.category_distribution
        assert "technical" in report.category_distribution
        assert report.category_distribution["general"] == 2
        assert report.category_distribution["technical"] == 1

    def test_source_doc_distribution(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        assert "doc1.pdf" in report.source_doc_distribution
        assert "doc2.pdf" in report.source_doc_distribution
        assert report.source_doc_distribution["doc1.pdf"] == 2
        assert report.source_doc_distribution["doc2.pdf"] == 1

    def test_answer_length_stats(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        stats = report.answer_length_stats
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "stdev" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_question_length_stats(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        stats = report.question_length_stats
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] > 0

    def test_imbalance_warning(self, analyzer):
        pairs = [
            QAPair(question="q", answer="a" * 20, source_doc="doc1.pdf", category="cat1")
        ] * 11 + [
            QAPair(question="q", answer="a" * 20, source_doc="doc2.pdf", category="cat1")
        ]
        report = analyzer.analyze(pairs)
        imbalance_warnings = [w for w in report.warnings if "불균형" in w]
        assert len(imbalance_warnings) > 0

    def test_insufficient_data_warning(self, analyzer, sample_pairs):
        report = analyzer.analyze(sample_pairs)
        data_warnings = [w for w in report.warnings if "적습니다" in w]
        assert len(data_warnings) > 0

    def test_single_category_warning(self, analyzer):
        pairs = [
            QAPair(question="q1?", answer="a" * 20, source_doc="doc1.pdf", category="only_one"),
            QAPair(question="q2?", answer="a" * 20, source_doc="doc1.pdf", category="only_one"),
        ]
        report = analyzer.analyze(pairs)
        cat_warnings = [w for w in report.warnings if "카테고리가 1개" in w]
        assert len(cat_warnings) > 0

    def test_save_report(self, analyzer, sample_pairs, tmp_path):
        report = analyzer.analyze(sample_pairs)
        output_path = tmp_path / "report.json"
        analyzer.save_report(report, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["total_pairs"] == 3
        assert "category_distribution" in data

    def test_is_augmented_count(self, analyzer, augmented_pairs):
        report = analyzer.analyze(augmented_pairs)
        assert report.total_pairs == 3
        assert report.original_pairs == 1
        assert report.augmented_pairs == 2

    def test_compute_stats_single_value(self, analyzer):
        stats = analyzer._compute_stats([42])
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["mean"] == 42.0
        assert stats["stdev"] == 0.0

    def test_compute_stats_empty(self, analyzer):
        stats = analyzer._compute_stats([])
        assert stats == {}
