"""HTML 파서(parsers/html.py)의 단위 테스트입니다."""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from slm_factory.parsers.html import HTMLParser, _table_to_markdown


# ---------------------------------------------------------------------------
# _table_to_markdown
# ---------------------------------------------------------------------------


class TestTableToMarkdown:
    """_table_to_markdown 함수의 테스트입니다."""

    def test_빈_테이블(self):
        """행이 없는 빈 테이블에서 빈 문자열을 반환하는지 확인합니다."""
        soup = BeautifulSoup("<table></table>", "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)
        assert result == ""

    def test_정상_테이블_변환(self):
        """헤더와 데이터 행이 있는 테이블을 마크다운으로 올바르게 변환하는지 확인합니다."""
        html = (
            "<table>"
            "<tr><th>이름</th><th>값</th></tr>"
            "<tr><td>A</td><td>1</td></tr>"
            "<tr><td>B</td><td>2</td></tr>"
            "</table>"
        )
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        result = _table_to_markdown(table)

        assert "| 이름 | 값 |" in result
        assert "| --- | --- |" in result
        assert "| A | 1 |" in result
        assert "| B | 2 |" in result


# ---------------------------------------------------------------------------
# HTMLParser
# ---------------------------------------------------------------------------


class TestHTMLParser:
    """HTMLParser 클래스의 테스트입니다."""

    def test_can_parse_html_True(self):
        """.html 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = HTMLParser()
        assert parser.can_parse(Path("page.html")) is True

    def test_can_parse_htm_True(self):
        """.htm 확장자에 대해 can_parse가 True를 반환하는지 확인합니다."""
        parser = HTMLParser()
        assert parser.can_parse(Path("page.htm")) is True

    def test_can_parse_pdf_False(self):
        """.pdf 확장자에 대해 can_parse가 False를 반환하는지 확인합니다."""
        parser = HTMLParser()
        assert parser.can_parse(Path("file.pdf")) is False

    def test_parse_기본_HTML(self, tmp_html_file):
        """기본 HTML 파일을 파싱하여 ParsedDocument를 반환하는지 확인합니다."""
        parser = HTMLParser()
        doc = parser.parse(tmp_html_file)

        assert doc.doc_id == "page.html"
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0

    def test_parse_제목_추출(self, tmp_html_file):
        """HTML의 <title> 태그에서 제목을 추출하는지 확인합니다."""
        parser = HTMLParser()
        doc = parser.parse(tmp_html_file)

        assert doc.title == "테스트 페이지"

    def test_parse_표_추출(self, tmp_html_file):
        """HTML의 <table>을 마크다운 표로 변환하여 tables에 저장하는지 확인합니다."""
        parser = HTMLParser()
        doc = parser.parse(tmp_html_file)

        assert len(doc.tables) >= 1
        # 마크다운 표 형식이 포함되어야 합니다
        assert "이름" in doc.tables[0]
        assert "|" in doc.tables[0]

    def test_parse_script_style_태그_제거(self, tmp_path):
        """script와 style 태그의 내용이 콘텐츠에서 제거되는지 확인합니다."""
        html_file = tmp_path / "script_test.html"
        html_file.write_text(
            "<html><body>"
            "<script>alert('위험!');</script>"
            "<style>body { color: red; }</style>"
            "<p>본문 텍스트입니다.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert "alert" not in doc.content
        assert "color: red" not in doc.content
        assert "본문 텍스트" in doc.content

    def test_parse_존재하지_않는_파일(self, tmp_path):
        """존재하지 않는 파일 경로에 대해 FileNotFoundError를 발생시키는지 확인합니다."""
        parser = HTMLParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.html")

    def test_parse_h1_폴백_제목(self, tmp_path):
        """<title>이 없으면 <h1>에서 제목을 추출하는지 확인합니다."""
        html_file = tmp_path / "no_title.html"
        html_file.write_text(
            "<html><body><h1>H1 제목</h1><p>내용</p></body></html>",
            encoding="utf-8",
        )

        parser = HTMLParser()
        doc = parser.parse(html_file)

        assert doc.title == "H1 제목"
