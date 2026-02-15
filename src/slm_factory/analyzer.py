"""학습 데이터 통계 분석 — QA 쌍의 품질을 수치로 보여줍니다."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import QAPair

from .utils import get_logger

logger = get_logger("analyzer")


@dataclass
class AnalysisReport:
    """QA 데이터 분석 결과를 담는 보고서입니다."""
    total_pairs: int = 0
    original_pairs: int = 0
    augmented_pairs: int = 0
    category_distribution: dict[str, int] = field(default_factory=dict)
    source_doc_distribution: dict[str, int] = field(default_factory=dict)
    answer_length_stats: dict[str, float] = field(default_factory=dict)
    question_length_stats: dict[str, float] = field(default_factory=dict)
    quality_score_stats: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class DataAnalyzer:
    """QA 쌍의 통계를 분석하여 데이터 품질을 수치로 보고합니다.
    
    LLM 의존성이 없으며 순수 계산만 수행합니다.
    """

    def analyze(self, pairs: list[QAPair]) -> AnalysisReport:
        """QA 쌍 리스트를 분석하여 AnalysisReport를 생성합니다."""
        report = AnalysisReport()
        
        if not pairs:
            report.warnings.append("데이터가 비어 있습니다.")
            return report
        
        report.total_pairs = len(pairs)
        report.original_pairs = sum(1 for p in pairs if not p.is_augmented)
        report.augmented_pairs = sum(1 for p in pairs if p.is_augmented)
        
        # 카테고리 분포
        categories = Counter(p.category or "uncategorized" for p in pairs)
        report.category_distribution = dict(categories.most_common())
        
        # 문서별 분포
        docs = Counter(p.source_doc or "unknown" for p in pairs)
        report.source_doc_distribution = dict(docs.most_common())
        
        # 답변 길이 통계
        answer_lengths = [len(p.answer) for p in pairs]
        report.answer_length_stats = self._compute_stats(answer_lengths)
        
        # 질문 길이 통계
        question_lengths = [len(p.question) for p in pairs]
        report.question_length_stats = self._compute_stats(question_lengths)
        
        # 경고 생성
        self._generate_warnings(report, pairs)
        
        return report
    
    def _compute_stats(self, values: Sequence[int | float]) -> dict[str, float]:
        """수치 리스트에서 기초 통계를 계산합니다."""
        if not values:
            return {}
        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
        }
    
    def _generate_warnings(self, report: AnalysisReport, pairs: list[QAPair]) -> None:
        """데이터 불균형이나 이상치를 경고합니다."""
        if report.source_doc_distribution:
            counts = list(report.source_doc_distribution.values())
            if max(counts) > 5 * min(counts) and min(counts) > 0:
                report.warnings.append(
                    f"문서별 QA 쌍 수 불균형이 심합니다 (최소: {min(counts)}, 최대: {max(counts)}). "
                    "특정 문서에 데이터가 편중되어 학습 편향이 발생할 수 있습니다."
                )
        
        if len(report.category_distribution) == 1:
            report.warnings.append(
                "카테고리가 1개뿐입니다. 다양한 카테고리를 추가하면 모델 성능이 향상됩니다."
            )
        
        if report.answer_length_stats.get("stdev", 0) > report.answer_length_stats.get("mean", 1) * 1.5:
            report.warnings.append(
                "답변 길이의 편차가 매우 큽니다. 일부 답변이 비정상적으로 길거나 짧을 수 있습니다."
            )
        
        if report.total_pairs < 50:
            report.warnings.append(
                f"학습 데이터가 {report.total_pairs}개로 적습니다. 최소 100개 이상을 권장합니다."
            )
    
    def print_summary(self, report: AnalysisReport) -> None:
        """Rich 콘솔에 분석 요약을 출력합니다."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        console.print(Panel("[bold cyan]학습 데이터 분석 보고서[/bold cyan]", expand=False))
        
        overview = Table(title="기본 통계", show_header=True)
        overview.add_column("항목", style="bold")
        overview.add_column("값", justify="right")
        overview.add_row("전체 QA 쌍", str(report.total_pairs))
        overview.add_row("원본 QA 쌍", str(report.original_pairs))
        overview.add_row("증강 QA 쌍", str(report.augmented_pairs))
        console.print(overview)
        
        if report.category_distribution:
            cat_table = Table(title="카테고리 분포", show_header=True)
            cat_table.add_column("카테고리", style="bold")
            cat_table.add_column("개수", justify="right")
            cat_table.add_column("비율", justify="right")
            for cat, count in report.category_distribution.items():
                ratio = f"{count / report.total_pairs * 100:.1f}%"
                cat_table.add_row(cat, str(count), ratio)
            console.print(cat_table)
        
        length_table = Table(title="길이 통계 (문자 수)", show_header=True)
        length_table.add_column("항목", style="bold")
        length_table.add_column("최소", justify="right")
        length_table.add_column("최대", justify="right")
        length_table.add_column("평균", justify="right")
        length_table.add_column("중앙값", justify="right")
        length_table.add_column("표준편차", justify="right")
        
        for label, stats in [("답변", report.answer_length_stats), ("질문", report.question_length_stats)]:
            if stats:
                length_table.add_row(
                    label,
                    str(int(stats["min"])),
                    str(int(stats["max"])),
                    str(stats["mean"]),
                    str(stats["median"]),
                    str(stats["stdev"]),
                )
        console.print(length_table)
        
        if report.warnings:
            console.print()
            for warning in report.warnings:
                console.print(f"[yellow]⚠ {warning}[/yellow]")
    
    def save_report(self, report: AnalysisReport, path: Path) -> None:
        """분석 보고서를 JSON 파일로 저장합니다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(report)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("분석 보고서를 %s에 저장했습니다.", path)
