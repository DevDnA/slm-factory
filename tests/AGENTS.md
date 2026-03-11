# tests/

39 test files, flat structure. pytest + pytest-mock + pytest-asyncio. No GPU required — ML libraries auto-mocked.

## STRUCTURE

Tests use underscore-flattened naming mirroring source modules:

| Test File | Source Module | Tests |
|-----------|---------------|-------|
| `test_cli.py` (875 lines) | `cli.py` | CliRunner integration (incl. compare-data) |
| `test_pipeline.py` (749 lines) | `pipeline.py` | step_* orchestration + regeneration |
| `test_integration.py` (622 lines) | cross-module | chunking→QA, ontology→QA, score→regen chains + relation dedup |
| `test_evolve_history.py` (703 lines) | `evolve_history.py` | version tracking |
| `test_config.py` | `config.py` | Pydantic validation (incl. ChunkingConfig, ScoringConfig.regenerate, RagConfig, AutoRAGExportConfig) |
| `test_autorag_export.py` (428 lines) | `exporter/autorag_export.py` | AutoRAG parquet export (corpus + QA chunking) |
| `test_device.py` | `device.py` | CUDA/MPS/CPU detection + multi-GPU |
| `test_trainer.py` | `trainer/lora_trainer.py` | DataLoader + LoRATrainer + DDP |
| `test_teacher.py` | `teacher/` | backend tests (Ollama streaming, OpenAI-compat) |
| `test_qa_generator.py` | `teacher/qa_generator.py` | QA generation + chunking |
| `test_ontology_*.py` (3 files) | `ontology/` | extractor, graph_store, models |
| `test_parsers_*.py` (6 files) | `parsers/*.py` | per-format parser tests (incl. HWP5, multi-section HWPX) |
| `test_dashboard.py` / `test_reviewer.py` | `tui/` | TUI widget tests |
| `test_exporter_gguf.py` (356 lines) | `exporter/` | GGUF 2-stage quantization export |
| Others (9 files) | Various | evaluator, scorer, augmenter, converter, etc. |

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
- `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`, `sentence_transformers`, `kiwipiepy`

This allows **all tests to run without GPU or ML dependencies installed**.

**Per-test hardware mocking** (test_device.py): `patch.dict("sys.modules", {"torch": mock_torch})` + `importlib.reload()` to test device detection without actual hardware.

## CONVENTIONS

- **Korean test names**: `def test_정상_문서_반환(self)` — descriptive Korean method names.
- **Korean docstrings**: Every test method has a Korean docstring.
- **Class-based**: All tests in classes (`class TestStepParse`), no bare functions.
- **Section separators**: `# ---...---` comment blocks between test groups.
- **1:1 mapping**: Each test file corresponds to one source module (except `test_integration.py` — cross-module chains).
- **Mock-heavy**: External deps (httpx, ML libs, file I/O) mocked via `mocker.patch()`.
- **CLI tests**: `typer.testing.CliRunner` → `runner.invoke(app, [...])` → assert exit code.
- **Local helpers**: Private `_make_*()` and `_mock_*()` functions per test file for domain-specific setup.
- **No test data directory**: All test data created inline via fixtures or `tmp_path`.

## ANTI-PATTERNS

- Do NOT add tests that require real GPU/ML libraries — maintain the mock strategy.
- Do NOT use bare test functions — always organize into classes.
- Do NOT skip the Korean naming convention — consistency matters.
- Do NOT add `conftest.py` in subdirectories — flat structure, single conftest.
