# parsers/

Document parsing subsystem. Registry pattern with ABC base class and 5 format-specific implementations.

## STRUCTURE

```
parsers/
├── __init__.py    # Registry instantiation + parser registration
├── base.py        # BaseParser ABC + ParserRegistry + shared helpers
├── pdf.py         # PDFParser — PyMuPDF (fitz), table extraction
├── hwpx.py        # HWPXParser — Korean HWPX (ZIP/XML), optional PyKoSpacing
├── html.py        # HTMLParser — BeautifulSoup4 + lxml, heading→markdown, charset-normalizer
├── text.py        # TextParser — plain text / markdown passthrough
└── docx.py        # DOCXParser — python-docx (optional dependency)
```

## HOW IT WORKS

1. `ParserRegistry` in `__init__.py` instantiates each parser class and stores instances.
2. `registry.get_parser(path)` matches file extension → returns first compatible parser.
3. `registry.parse_directory(dir, formats)` batch-parses with Rich progress bar; failures logged and skipped.
4. All parsers return `ParsedDocument` (from `models.py`): `doc_id`, `title`, `content` (markdown), `tables`, `metadata`.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new format parser | New file + `__init__.py` | Subclass `BaseParser`, set `extensions`, implement `parse()`, register |
| Change table rendering | `base.py:rows_to_markdown()` | Shared by PDF, HWPX, HTML, DOCX |
| Change date extraction | `base.py:extract_date_from_filename()` | YYMMDD pattern from filenames |
| Fix PDF table parsing | `pdf.py` | Uses `fitz.page.find_tables()` — `type: ignore` for untyped PyMuPDF |
| Fix Korean spacing | `hwpx.py` | Optional `pykospacing` dep — graceful fallback |
| Change encoding detection | `base.py:detect_encoding()` | Shared by HTML and Text parsers. Uses charset-normalizer for EUC-KR/CP949 |

## CONVENTIONS

- `extensions: ClassVar[list[str]]` declares supported file types per parser.
- Output content is always **markdown-formatted** — headings, paragraphs, tables.
- `metadata` dict holds parser-specific extras (page count, dates, authors).
- Optional deps use `try/except ImportError` with `None` fallback (see DOCX in `__init__.py`).
- `type: ignore` on PyMuPDF calls (`pdf.py:31,80`) — library has no type stubs.
- `detect_encoding()` in `base.py` is the shared encoding detection utility — do NOT duplicate in individual parsers.

## ANTI-PATTERNS

- Do NOT add parsers without registering them in `__init__.py`.
- Do NOT return raw text — always markdown-format the output.
- The `parse()` method must never raise on recoverable errors — log and return partial results.
