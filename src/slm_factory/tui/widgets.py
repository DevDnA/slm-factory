"""QA 리뷰 TUI 위젯입니다."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    from ..models import QAPair

_STATUS_BADGE = {
    "approved": "[bold green]✓ 승인[/bold green]",
    "rejected": "[bold red]✗ 거부[/bold red]",
}
_PENDING_BADGE = "[bold yellow]? 대기[/bold yellow]"


class QACard(Static):
    """단일 QA 쌍을 Rich 마크업으로 표시하는 위젯입니다."""

    CSS = """
    QACard {
        border: solid $accent;
        padding: 1 2;
        margin: 1 2;
        height: auto;
    }
    """

    def update_pair(self, pair: QAPair) -> None:
        """QA 쌍 내용으로 위젯을 갱신합니다."""
        badge = _STATUS_BADGE.get(pair.review_status, _PENDING_BADGE)
        source = pair.source_doc or "-"
        category = pair.category or "-"

        content = (
            f"{badge}\n\n"
            f"[bold cyan]📂 출처:[/bold cyan] {source}  │  "
            f"[bold cyan]📁 분류:[/bold cyan] {category}\n\n"
            f"[bold]❓ 질문[/bold]\n{pair.question}\n\n"
            f"[bold]💡 답변[/bold]\n{pair.answer}"
        )
        self.update(content)


class StatusBar(Static):
    """현재 위치와 승인/거부/대기 건수를 표시하는 상태 바 위젯입니다."""

    CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 2;
    }
    """

    def update_status(
        self,
        current: int,
        total: int,
        approved: int,
        rejected: int,
        pending: int,
    ) -> None:
        """상태 바 내용을 갱신합니다."""
        self.update(
            f" [{current}/{total}]  "
            f"[green]승인 {approved}[/green]  "
            f"[red]거부 {rejected}[/red]  "
            f"[yellow]대기 {pending}[/yellow]"
        )
