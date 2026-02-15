"""파서 기본 ABC, ParsedDocument 데이터클래스, ParserRegistry 정의."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from ..models import ParsedDocument
from ..utils import get_logger

logger = get_logger("parsers.base")

# 파일명에서 자주 발견되는 YYMMDD 날짜 패턴 (예: "report_240115_v2.pdf")
_DATE_PATTERN = re.compile(r"(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")


def extract_date_from_filename(filename: str) -> str | None:
    """파일명에서 YYMMDD 날짜를 추출하여 YYYY-MM-DD 형식으로 반환합니다.

    2000년대 세기를 가정합니다. 유효한 패턴이 없으면 None을 반환합니다.
    """
    match = _DATE_PATTERN.search(filename)
    if match:
        yy, mm, dd = match.groups()
        return f"20{yy}-{mm}-{dd}"
    return None


class BaseParser(ABC):
    """모든 문서 파서의 추상 기본 클래스입니다."""

    extensions: ClassVar[list[str]] = []
    """이 파서가 처리하는 파일 확장자입니다 (예: ['.pdf'])."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """*path*의 문서를 파싱하여 ParsedDocument를 반환합니다."""
        ...

    def can_parse(self, path: Path) -> bool:
        """이 파서가 주어진 파일 확장자를 지원하면 True를 반환합니다."""
        return path.suffix.lower() in self.extensions


class ParserRegistry:
    """파일 확장자로 파서를 자동 발견하고 선택합니다.

    사용 예::

        registry = ParserRegistry()

        @registry.register
        class MyParser(BaseParser):
            extensions = ['.xyz']
            def parse(self, path): ...

        doc = registry.get_parser(Path('file.xyz')).parse(Path('file.xyz'))
        docs = registry.parse_directory(Path('./docs'), formats=['.pdf'])
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    # ------------------------------------------------------------------
    # 등록
    # ------------------------------------------------------------------

    def register(self, parser_cls: type[BaseParser]) -> type[BaseParser]:
        """파서 클래스를 등록합니다 (데코레이터로 사용 가능).

        클래스를 인스턴스화하고 인스턴스를 저장합니다.

        클래스를 변경하지 않고 반환하므로 직접 사용할 수 있습니다.
        """
        instance = parser_cls()
        self._parsers.append(instance)
        logger.debug("Registered parser %s for %s", parser_cls.__name__, parser_cls.extensions)
        return parser_cls

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def get_parser(self, path: Path) -> BaseParser | None:
        """*path*를 처리할 수 있는 첫 번째 등록된 파서를 반환하거나 None을 반환합니다."""
        for parser in self._parsers:
            if parser.can_parse(path):
                return parser
        return None

    # ------------------------------------------------------------------
    # 배치 파싱
    # ------------------------------------------------------------------

    def parse_directory(
        self,
        dir_path: Path,
        formats: list[str] | None = None,
        files: list[Path] | None = None,
    ) -> list[ParsedDocument]:
        """*dir_path*의 모든 지원 파일을 파싱합니다 (재귀 없음).

        매개변수
        ----------
        dir_path:
            스캔할 디렉토리입니다 (*files* 미지정 시 사용).
        formats:
            확장자의 선택적 화이트리스트입니다 (예: ``['.pdf']``).
            *None*일 때, 모든 등록된 확장자가 허용됩니다.
        files:
            파싱할 파일 목록입니다. 지정 시 *dir_path*와 *formats*를 무시하고
            이 목록의 파일만 파싱합니다.

        반환값
        -------
        list[ParsedDocument]
            성공적으로 파싱된 문서입니다 (실패는 로깅되고 건너뜁니다).
        """
        if files is not None:
            target_files: list[Path] = [Path(f) for f in files]
        else:
            dir_path = Path(dir_path)
            if not dir_path.is_dir():
                logger.error("Directory not found: %s", dir_path)
                return []

            allowed = {ext.lower() for ext in formats} if formats else None
            target_files = sorted(
                f
                for f in dir_path.iterdir()
                if f.is_file()
                and self.get_parser(f) is not None
                and (allowed is None or f.suffix.lower() in allowed)
            )

        if not target_files:
            logger.warning("No parseable files found in %s", dir_path)
            return []

        documents: list[ParsedDocument] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Parsing documents", total=len(target_files))
            for file_path in target_files:
                parser = self.get_parser(file_path)
                if parser is None:
                    progress.advance(task)
                    continue
                try:
                    doc = parser.parse(file_path)
                    documents.append(doc)
                    logger.debug("Parsed: %s", file_path.name)
                except Exception:
                    logger.exception("Failed to parse %s", file_path.name)
                finally:
                    progress.advance(task)

        logger.info("Parsed %d / %d files from %s", len(documents), len(target_files), dir_path)
        return documents
