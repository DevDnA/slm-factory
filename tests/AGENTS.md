# tests/

30 test files, flat structure. pytest + pytest-mock + pytest-asyncio. No GPU required — ML libraries auto-mocked.

## STRUCTURE

Tests use underscore-flattened naming mirroring source modules:

| Test File | Source Module | Tests |
|-----------|---------------|-------|
| `test_cli.py` | `cli.py` | 45 — CliRunner integration |
| `test_pipeline.py` | `pipeline.py` | 27 — step_* orchestration |
| `test_evolve_history.py` | `evolve_history.py` | 41 — version tracking |
| `test_dashboard.py` | `tui/dashboard.py` | 29 — TUI dashboard |
| `test_reviewer.py` | `tui/reviewer.py` | 28 — TUI reviewer |
| `test_config.py` | `config.py` | 26 — Pydantic validation |
| `test_incremental.py` | `incremental.py` | 24 — hash tracking |
| `test_parsers_*.py` (6 files) | `parsers/*.py` | ~92 — per-format parser tests |
| `test_teacher.py` | `teacher/` | 16 — backend tests |
| `test_qa_generator.py` | `teacher/qa_generator.py` | 16 — QA generation |
| `test_dialogue_generator.py` | `teacher/dialogue_generator.py` | 20 — dialogue gen |
| Others (8 files) | Various | ~90 — evaluator, scorer, augmenter, etc. |

## KEY FIXTURES (`conftest.py`)

| Fixture | Returns | Usage |
|---------|---------|-------|
| `make_config` | `(**overrides) → SLMConfig` | Factory — pass overrides for any sub-config |
| `make_qa_pair` | `(question=..., answer=...) → QAPair` | Factory — quick QAPair creation |
| `make_parsed_doc` | `(doc_id=..., content=...) → ParsedDocument` | Factory — quick doc creation |
| `default_config` | Pre-built `SLMConfig` | Default config for simple tests |
| `sample_validation_config` | Pre-built `ValidationConfig` | Validation-specific tests |
| `tmp_text_file` / `tmp_md_file` / `tmp_html_file` | `Path` | Temp files via `tmp_path` |
| `tmp_yaml_config` | `Path` | Temp project.yaml |

## ML MOCKING STRATEGY

`conftest.py` calls `_ensure_ml_mocks()` at module load time, which registers `MagicMock` objects for heavy ML libraries into `sys.modules`:

- `torch`, `torch.cuda`, `torch.backends`, `torch.nn`, etc.
- `transformers`, `transformers.AutoModelForCausalLM`, `transformers.AutoTokenizer`, etc.
- `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`, `sentence_transformers`

This allows **all tests to run without GPU or ML dependencies installed**.

## CONVENTIONS

- **Korean test names**: `def test_정상_문서_반환(self)` — descriptive Korean method names.
- **Korean docstrings**: Every test method has a Korean docstring.
- **Class-based**: All tests in classes (`class TestStepParse`), no bare functions.
- **Section separators**: `# ---...---` comment blocks between test groups.
- **1:1 mapping**: Each test file corresponds to one source module.
- **Mock-heavy**: External deps (httpx, ML libs, file I/O) mocked via `mocker.patch()`.
- **CLI tests**: `typer.testing.CliRunner` → `runner.invoke(app, [...])` → assert exit code.
- **Local helpers**: Private `_make_*()` and `_mock_*()` functions per test file for domain-specific setup.
- **No test data directory**: All test data created inline via fixtures or `tmp_path`.

## ANTI-PATTERNS

- Do NOT add tests that require real GPU/ML libraries — maintain the mock strategy.
- Do NOT use bare test functions — always organize into classes.
- Do NOT skip the Korean naming convention — consistency matters.
- Do NOT add `conftest.py` in subdirectories — flat structure, single conftest.
