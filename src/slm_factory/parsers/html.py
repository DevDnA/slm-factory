"""HTML 파서 — BeautifulSoup을 사용하여 .html/.htm 파일에서 텍스트를 추출합니다."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup

from ..utils import get_logger
from .base import BaseParser, ParsedDocument, extract_date_from_filename

logger = get_logger("parsers.html")


def _detect_encoding(content: bytes) -> str:
    """콘텐츠에서 인코딩을 감지하고, 폴백은 latin-1입니다."""
    # 먼저 UTF-8 시도
    try:
        content.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # 메타 태그에서 charset 찾기 시도
    try:
        soup = BeautifulSoup(content, "html.parser")
        meta_charset = soup.find("meta", charset=True)
        if meta_charset:
            return meta_charset.get("charset", "latin-1")
    except Exception:
        pass

    # 폴백
    return "latin-1"


def _table_to_markdown(table_element) -> str:
    """HTML 표 요소를 마크다운 형식으로 변환합니다."""
    try:
        rows = table_element.find_all("tr")
        if not rows:
            return ""

        table_rows = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            row_data = []
            for cell in cells:
                cell_text = cell.get_text(separator=" ", strip=True)
                row_data.append(cell_text)
            if row_data:
                table_rows.append(row_data)

        if not table_rows:
            return ""

        # 마크다운 표 구성
        header = table_rows[0]
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in table_rows[1:]:
            # 행을 헤더 길이에 맞게 패딩
            while len(row) < len(header):
                row.append("")
            md_lines.append("| " + " | ".join(row[:len(header)]) + " |")

        return "\n".join(md_lines)
    except Exception:
        logger.debug("Failed to convert table to markdown", exc_info=True)
        return ""


class HTMLParser(BaseParser):
    """HTML 문서를 파싱합니다."""

    extensions: ClassVar[list[str]] = [".html", ".htm"]

    def parse(self, path: Path) -> ParsedDocument:
        """HTML 파일에서 텍스트와 표를 추출합니다.

        매개변수
        ----------
        path:
            HTML 파일의 경로입니다.

        반환값
        -------
        ParsedDocument
            텍스트 내용, 표 (마크다운), 메타데이터로 채워집니다.

        예외
        ------
        FileNotFoundError
            *path*가 존재하지 않으면 발생합니다.
        RuntimeError
            HTML 파일을 파싱할 수 없으면 발생합니다.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")

        # ------------------------------------------------------------------
        # 인코딩 감지로 파일 읽기
        # ------------------------------------------------------------------
        try:
            raw_content = path.read_bytes()
            encoding = _detect_encoding(raw_content)
            html_content = raw_content.decode(encoding)
        except Exception as exc:
            raise RuntimeError(f"Failed to read HTML file: {path}") from exc

        # ------------------------------------------------------------------
        # HTML 파싱
        # ------------------------------------------------------------------
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse HTML: {path}") from exc

        # ------------------------------------------------------------------
        # 텍스트 추출 전에 script 및 style 태그 제거
        # ------------------------------------------------------------------
        for tag in soup.find_all(["script", "style", "nav"]):
            tag.decompose()

        # ------------------------------------------------------------------
        # 제목 추출
        # ------------------------------------------------------------------
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        if not title:
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text(strip=True)

        if not title:
            title = path.stem

        # ------------------------------------------------------------------
        # 주요 텍스트 추출
        # ------------------------------------------------------------------
        content = soup.get_text(separator="\n", strip=True)

        # 과도한 빈 줄 정규화
        content = re.sub(r"\n{3,}", "\n\n", content)

        # ------------------------------------------------------------------
        # 표 추출
        # ------------------------------------------------------------------
        tables_md: list[str] = []
        for table in soup.find_all("table"):
            md = _table_to_markdown(table)
            if md:
                tables_md.append(md)

        # ------------------------------------------------------------------
        # 메타데이터
        # ------------------------------------------------------------------
        metadata: dict[str, object] = {}

        # 파일명에서 날짜 추출 시도 (YYMMDD)
        date_from_name = extract_date_from_filename(path.stem)
        if date_from_name:
            metadata["date"] = date_from_name

        return ParsedDocument(
            doc_id=path.name,
            title=title,
            content=content,
            tables=tables_md,
            metadata=metadata,
        )
