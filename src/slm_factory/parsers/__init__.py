"""다양한 파일 형식의 문서 파서입니다."""

from .base import BaseParser, ParsedDocument, ParserRegistry
from .pdf import PDFParser
from .hwpx import HWPXParser
from .html import HTMLParser
from .text import TextParser

__all__ = [
    "BaseParser",
    "ParsedDocument",
    "ParserRegistry",
    "PDFParser",
    "HWPXParser",
    "HTMLParser",
    "TextParser",
]

registry = ParserRegistry()
registry.register(PDFParser)
registry.register(HWPXParser)
registry.register(HTMLParser)
registry.register(TextParser)
