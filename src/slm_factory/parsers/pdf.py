"""PyMuPDF (fitz)를 사용한 PDF 파서 — 빠른 네이티브 텍스트 및 표 추출."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import fitz  # PyMuPDF

from ..utils import get_logger
from .base import BaseParser, ParsedDocument, extract_date_from_filename, rows_to_markdown

logger = get_logger("parsers.pdf")

# 제거할 페이지 번호 라인의 패턴 (예: "- 1 -", "— 3 —", 독립형 숫자)
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*[-—–]\s*\d+\s*[-—–]\s*$", re.MULTILINE),  # - 1 -, — 12 —
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),                    # 독립형 페이지 번호
    re.compile(r"^\s*page\s+\d+\s*$", re.MULTILINE | re.IGNORECASE),  # "Page 3"
]


def _clean_page_numbers(text: str) -> str:
    """추출된 텍스트에서 일반적인 페이지 번호 패턴을 제거합니다."""
    for pattern in _PAGE_NUM_PATTERNS:
        text = pattern.sub("", text)
    return text


def _table_to_markdown(table: fitz.table.Table) -> str:  # type: ignore[name-defined]
    """PyMuPDF Table 객체를 마크다운 표 문자열로 변환합니다."""
    try:
        raw_rows = table.extract()
    except Exception:
        return ""

    if not raw_rows:
        return ""

    def _cell(val: object) -> str:
        if val is None:
            return ""
        return str(val).replace("\n", " ").strip()

    table_rows = [[_cell(c) for c in row] for row in raw_rows]
    return rows_to_markdown(table_rows)


class PDFParser(BaseParser):
    """PyMuPDF (fitz)를 사용하여 PDF 문서를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".pdf"]

    def parse(self, path: Path) -> ParsedDocument:
        """PDF 파일에서 텍스트와 표를 추출합니다.

        매개변수
        ----------
        path:
            PDF 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            PyMuPDF가 파일을 열 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            doc = fitz.open(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(
                f"PDF 파일을 열 수 없습니다: {path}\n"
                f"원인: 파일이 손상되었거나, 암호화되었거나, 지원하지 않는 형식일 수 있습니다.\n"
                f"해결: 다른 PDF 뷰어에서 열어 정상 파일인지 확인하세요."
            ) from exc

        # ------------------------------------------------------------------
        # 텍스트 추출
        # ------------------------------------------------------------------
        pages_text: list[str] = []
        for page in doc:
            try:
                text = page.get_text("text")
                if text:
                    pages_text.append(text)
            except Exception:
                logger.warning("Could not extract text from page %d of %s", page.number, path.name)

        raw_text = "\n\n".join(pages_text)
        content = _clean_page_numbers(raw_text).strip()

        # 과도한 빈 줄 정규화
        content = re.sub(r"\n{3,}", "\n\n", content)

        # ------------------------------------------------------------------
        # 표 추출 (최선 노력; PyMuPDF >= 1.23.0은 find_tables 포함)
        # ------------------------------------------------------------------
        tables_md: list[str] = []
        try:
            for page in doc:
                tab_finder = page.find_tables()
                for table in tab_finder.tables:
                    md = _table_to_markdown(table)
                    if md:
                        tables_md.append(md)
        except AttributeError:
            # find_tables 없는 PyMuPDF 버전 — 조용히 건너뜀
            logger.debug("Table extraction unavailable (PyMuPDF version lacks find_tables)")
        except Exception:
            logger.debug("Table extraction failed for %s", path.name, exc_info=True)

        # ------------------------------------------------------------------
        # 메타데이터
        # ------------------------------------------------------------------
        pdf_meta = doc.metadata or {}
        metadata: dict[str, object] = {}

        if pdf_meta.get("author"):
            metadata["author"] = pdf_meta["author"]
        if pdf_meta.get("creationDate"):
            metadata["creation_date"] = pdf_meta["creationDate"]
        if pdf_meta.get("modDate"):
            metadata["mod_date"] = pdf_meta["modDate"]
        if pdf_meta.get("subject"):
            metadata["subject"] = pdf_meta["subject"]

        metadata["page_count"] = len(doc)

        # 파일명에서 날짜 추출 시도 (YYMMDD)
        date_from_name = extract_date_from_filename(path.stem)
        if date_from_name:
            metadata["date"] = date_from_name

        # ------------------------------------------------------------------
        # 제목: PDF 메타데이터 선호, 파일명으로 폴백
        # ------------------------------------------------------------------
        title = (pdf_meta.get("title") or "").strip()
        if not title:
            title = path.stem

        doc.close()

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )
