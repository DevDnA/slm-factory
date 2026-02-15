"""파이프라인 모니터링 TUI 대시보드입니다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
    from ..config import DashboardConfig, SLMConfig

from ..utils import get_logger

logger = get_logger("dashboard")


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------


@dataclass
class StageInfo:
    """파이프라인 단계 정보입니다."""

    name: str
    display_name: str
    filename: str
    exists: bool = False
    count: int = 0
    unit: str = "건"


@dataclass
class PipelineSnapshot:
    """파이프라인 상태 스냅샷입니다."""

    stages: list[StageInfo] = field(default_factory=list)
    eval_summary: dict[str, object] = field(default_factory=dict)
    compare_summary: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 파이프라인 스캔
# ---------------------------------------------------------------------------

_STAGE_DEFS: list[tuple[str, str, str, str]] = [
    ("parse", "문서 파싱", "parsed_documents.json", "문서"),
    ("generate", "QA 생성", "qa_alpaca.json", "쌍"),
    ("score", "품질 평가", "qa_scored.json", "쌍"),
    ("augment", "데이터 증강", "qa_augmented.json", "쌍"),
    ("analyze", "데이터 분석", "data_analysis.json", "항목"),
    ("convert", "학습 변환", "training_data.jsonl", "줄"),
    ("train", "모델 학습", "checkpoints/adapter/", "디렉토리"),
    ("export", "모델 병합", "merged_model/", "디렉토리"),
    ("eval", "모델 평가", "eval_results.json", "건"),
    ("compare", "모델 비교", "compare_results.json", "건"),
    ("dialogue", "대화 생성", "dialogues.json", "건"),
]


def scan_pipeline(output_dir: Path) -> PipelineSnapshot:
    """출력 디렉토리를 스캔하여 파이프라인 상태 스냅샷을 반환합니다.

    매개변수
    ----------
    output_dir:
        파이프라인 출력 디렉토리 경로입니다.

    반환값
    -------
    PipelineSnapshot
        현재 파이프라인 상태를 담은 스냅샷입니다.
    """
    stages: list[StageInfo] = []

    for name, display, filename, unit in _STAGE_DEFS:
        filepath = output_dir / filename.rstrip("/")
        info = StageInfo(
            name=name,
            display_name=display,
            filename=filename,
            unit=unit,
        )

        if filename.endswith("/"):
            # 디렉토리 존재 여부 확인
            info.exists = filepath.is_dir()
        elif filename.endswith(".jsonl"):
            # JSONL 파일 줄 수 카운트
            if filepath.is_file():
                info.exists = True
                info.count = sum(1 for _ in filepath.open(encoding="utf-8"))
        elif filepath.is_file():
            # JSON 파일 항목 카운트
            info.exists = True
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                info.count = len(data) if isinstance(data, list) else 1
            except Exception:
                info.count = 0
        stages.append(info)

    # 평가 결과 요약 추출
    eval_summary = _extract_eval_summary(output_dir / "eval_results.json")
    compare_summary = _extract_compare_summary(output_dir / "compare_results.json")

    return PipelineSnapshot(
        stages=stages,
        eval_summary=eval_summary,
        compare_summary=compare_summary,
    )


def _extract_eval_summary(path: Path) -> dict[str, object]:
    """eval_results.json에서 평균 점수를 추출합니다.

    매개변수
    ----------
    path:
        eval_results.json 파일 경로입니다.

    반환값
    -------
    dict[str, object]
        메트릭 이름을 키로, 평균 점수를 값으로 가지는 딕셔너리입니다.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, list) or not data:
        return {}

    # 각 결과 항목의 scores 필드에서 메트릭별 평균을 계산합니다
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in data:
        scores = item.get("scores", {}) if isinstance(item, dict) else {}
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                totals[metric] = totals.get(metric, 0.0) + value
                counts[metric] = counts.get(metric, 0) + 1

    return {
        metric: round(totals[metric] / counts[metric], 4)
        for metric in totals
        if counts.get(metric, 0) > 0
    }


def _extract_compare_summary(path: Path) -> dict[str, object]:
    """compare_results.json에서 비교 요약을 추출합니다.

    매개변수
    ----------
    path:
        compare_results.json 파일 경로입니다.

    반환값
    -------
    dict[str, object]
        비교 결과 요약 딕셔너리입니다.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(data, dict):
        summary = data.get("summary")
        if isinstance(summary, dict) and summary:
            return dict(summary)
        return {"total": len(data)}

    if isinstance(data, list):
        return {"total": len(data)}

    return {}


# ---------------------------------------------------------------------------
# TUI 위젯
# ---------------------------------------------------------------------------


class StageTable(Static):
    """파이프라인 단계를 표 형태로 표시하는 위젯입니다."""

    def update_snapshot(self, snapshot: PipelineSnapshot) -> None:
        """스냅샷 데이터로 테이블을 갱신합니다.

        매개변수
        ----------
        snapshot:
            파이프라인 상태 스냅샷입니다.
        """
        from rich.table import Table

        table = Table(title="파이프라인 진행 상태", expand=True)
        table.add_column("단계", style="cyan", ratio=1)
        table.add_column("파일", style="dim", ratio=2)
        table.add_column("상태", style="bold", justify="center", ratio=1)
        table.add_column("건수", justify="right", ratio=1)

        for stage in snapshot.stages:
            status = "[green]✓[/green]" if stage.exists else "[red]✗[/red]"
            if stage.exists:
                if stage.filename.endswith("/"):
                    count_str = stage.unit
                else:
                    count_str = f"{stage.count}개 {stage.unit}"
            else:
                count_str = "-"

            table.add_row(stage.display_name, stage.filename, status, count_str)

        self.update(table)


class MetricsSummary(Static):
    """평가 메트릭 및 비교 결과를 표시하는 위젯입니다."""

    def update_snapshot(self, snapshot: PipelineSnapshot) -> None:
        """스냅샷 데이터로 메트릭 요약을 갱신합니다.

        매개변수
        ----------
        snapshot:
            파이프라인 상태 스냅샷입니다.
        """
        from rich.panel import Panel
        from rich.text import Text

        sections: list[str] = []

        if snapshot.eval_summary:
            lines = ["[bold cyan]📊 평가 결과[/bold cyan]"]
            for metric, value in snapshot.eval_summary.items():
                lines.append(f"  {metric}: {value}")
            sections.append("\n".join(lines))

        if snapshot.compare_summary:
            lines = ["[bold cyan]🔀 비교 결과[/bold cyan]"]
            for key, value in snapshot.compare_summary.items():
                lines.append(f"  {key}: {value}")
            sections.append("\n".join(lines))

        # 데이터 분석 통계 (data_analysis.json 단계 정보)
        analysis_stage = next(
            (s for s in snapshot.stages if s.name == "analyze" and s.exists),
            None,
        )
        if analysis_stage:
            sections.append(
                f"[bold cyan]📈 데이터 분석[/bold cyan]\n"
                f"  분석 항목: {analysis_stage.count}개"
            )

        if sections:
            content = "\n\n".join(sections)
        else:
            content = "[dim]평가·비교 결과가 아직 없습니다.[/dim]"

        self.update(Panel(content, title="메트릭 요약", expand=True))


# ---------------------------------------------------------------------------
# 대시보드 앱
# ---------------------------------------------------------------------------


class PipelineDashboard(App[None]):
    """파이프라인 모니터링 TUI 대시보드입니다."""

    CSS = """
    StageTable {
        height: auto;
        margin: 1 2;
    }
    MetricsSummary {
        height: auto;
        margin: 1 2;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "새로고침"),
        Binding("q", "quit", "종료"),
    ]

    def __init__(
        self,
        output_dir: Path,
        refresh_interval: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """대시보드를 초기화합니다.

        매개변수
        ----------
        output_dir:
            파이프라인 출력 디렉토리 경로입니다.
        refresh_interval:
            자동 새로고침 간격(초)입니다.
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.refresh_interval = refresh_interval

    def compose(self) -> ComposeResult:
        """대시보드 레이아웃을 구성합니다."""
        yield Header(show_clock=True)
        with Vertical():
            yield StageTable()
            yield MetricsSummary()
        yield Footer()

    def on_mount(self) -> None:
        """마운트 시 초기 스캔 및 자동 새로고침을 설정합니다."""
        self.action_refresh()
        self.set_interval(self.refresh_interval, self.action_refresh)

    def action_refresh(self) -> None:
        """출력 디렉토리를 다시 스캔하여 화면을 갱신합니다."""
        snapshot = scan_pipeline(self.output_dir)
        self.query_one(StageTable).update_snapshot(snapshot)
        self.query_one(MetricsSummary).update_snapshot(snapshot)
