"""다양한 파일 형식의 문서 파서입니다."""

from .base import BaseParser, ParsedDocument, ParserRegistry
from .pdf import PDFParser
from .hwpx import HWPXParser
from .html import HTMLParser
from .text import TextParser

# DOCX 파서는 선택적 종속성입니다
try:
    from .docx import DOCXParser  # noqa: F401
except ImportError:
    DOCXParser = None  # type: ignore

__all__ = [
    "BaseParser",
    "ParsedDocument",
    "ParserRegistry",
    "PDFParser",
    "HWPXParser",
    "HTMLParser",
    "TextParser",
    "DOCXParser",
]

registry = ParserRegistry()
registry.register(PDFParser)
registry.register(HWPXParser)
registry.register(HTMLParser)
registry.register(TextParser)

if DOCXParser is not None:
    registry.register(DOCXParser)
