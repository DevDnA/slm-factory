# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-09
**Commit:** c8f67c8
**Branch:** main

## OVERVIEW

Teacher-Student knowledge distillation CLI framework for building domain-specific Small Language Models (SLMs) from documents. Korean-authored. Python 3.11+, Typer CLI, Pydantic v2 config, PyTorch/HuggingFace ML stack, Ollama deployment.

## STRUCTURE

```
slm-factory/
├── src/slm_factory/          # Source (src layout, setuptools)
│   ├── cli.py                # ALL CLI commands — monolith (2202 lines)
│   ├── pipeline.py           # Orchestrator — 12 step_* methods + async regeneration
│   ├── config.py             # 30 Pydantic v2 models for YAML config (incl. ChunkingConfig)
│   ├── models.py             # Shared dataclasses: QAPair, ParsedDocument, etc.
│   ├── device.py             # CUDA/MPS/CPU auto-detection + multi-GPU (gpu_count)
│   ├── utils.py              # Logging (Rich), async helpers, hash utils
│   ├── analyzer.py           # QA data statistics
│   ├── augmenter.py          # Paraphrase augmentation via Teacher LLM
│   ├── comparator.py         # Base vs fine-tuned model comparison
│   ├── converter.py          # QA → chat-template JSONL
│   ├── evaluator.py          # BLEU/ROUGE evaluation (Korean morpheme-based)
│   ├── evolve_history.py     # Versioned model evolution tracking
│   ├── incremental.py        # Document change detection (hash-based)
│   ├── scorer.py             # LLM-based QA quality scoring
│   ├── parsers/              # → see parsers/AGENTS.md
│   ├── teacher/              # → see teacher/AGENTS.md
│   ├── exporter/             # → see exporter/AGENTS.md
│   ├── trainer/              # → see trainer/AGENTS.md
│   ├── validator/            # → see validator/AGENTS.md
│   ├── ontology/             # → see ontology/AGENTS.md
│   └── tui/                  # → see tui/AGENTS.md
├── tests/                    # → see tests/AGENTS.md
├── templates/project.yaml    # Default config template (NOT in package)
├── docs/                     # 6 Korean markdown guides
└── pyproject.toml            # Single build config (setuptools)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add CLI command | `cli.py` | Typer `@app.command()` or `@tool_app.command()` |
| Add pipeline step | `pipeline.py` + `config.py` | New `step_*` method + config model + `SLMConfig` field + `_STEP_ORDER` |
| Add document parser | `parsers/` | Subclass `BaseParser`, register in `__init__.py` |
| Add teacher backend | `teacher/` | Subclass `BaseTeacher`, add to `create_teacher()` factory |
| Add export format | `exporter/` | New exporter class, wire in `pipeline.step_export()` |
| Change QA data shape | `models.py` | `QAPair` is used by 19+ modules — ripple risk |
| Change config schema | `config.py` | Add Pydantic model + field on `SLMConfig` |
| Hardware-specific logic | `device.py` | `DeviceInfo` + `detect_device()` + `get_training_overrides()` |
| Error hint for CLI | `cli.py:_get_error_hints()` | Pattern-matches exceptions → Korean help text |
| Test setup | `tests/conftest.py` | Factory fixtures + ML mocking |
| Default project config | `templates/project.yaml` | Template for `slm-factory init` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Pipeline` | class | `pipeline.py:23` | Central orchestrator — 12 `step_*` methods |
| `Pipeline._regenerate_round` | async | `pipeline.py` | Async batch regeneration per round (semaphore + gather) |
| `SLMConfig` | class | `config.py:430` | Root Pydantic v2 config (20 sub-configs) |
| `ChunkingConfig` | class | `config.py:431` | Document chunking settings (chunk_size, overlap_chars) |
| `load_config` | func | `config.py:474` | YAML → validated `SLMConfig` |
| `app` | Typer | `cli.py:30` | CLI entry point (`slm-factory` command) |
| `_STEP_ORDER` | list | `cli.py:393` | Pipeline step execution order (9 steps) |
| `QAPair` | dataclass | `models.py:29` | Shared data model — system's lingua franca |
| `ParsedDocument` | dataclass | `models.py:9` | Document parser output |
| `BaseParser` | ABC | `parsers/base.py:81` | Parser interface — `parse(path) → ParsedDocument` |
| `ParserRegistry` | class | `parsers/base.py:97` | Registration + batch parsing with Rich progress |
| `detect_encoding` | func | `parsers/base.py:22` | charset-normalizer encoding detection (EUC-KR/CP949/UTF-8) |
| `BaseTeacher` | ABC | `teacher/base.py:8` | LLM interface — `generate()` / `agenerate()` |
| `create_teacher` | func | `teacher/__init__.py:22` | Factory: config.backend → teacher instance |
| `DeviceInfo` | dataclass | `device.py:19` | Frozen hardware detection (incl. `gpu_count` for multi-GPU) |
| `detect_device` | func | `device.py:86` | CUDA → MPS → CPU priority detection |
| `run_async` | func | `utils.py` | Async bridge — handles event loop detection + nest_asyncio |
| `run_bounded` | func | `utils.py` | Semaphore-bounded async execution with progress tracking |
| `EvolveHistory` | class | `evolve_history.py` | Version tracking + quality gate + cleanup |
| `IncrementalTracker` | class | `incremental.py` | Document hash-based change detection |
| `OntologyExtractor` | class | `ontology/extractor.py` | LLM-based entity/relation extraction from documents |
| `KnowledgeGraph` | dataclass | `ontology/models.py` | Entity + Relation container with context string export |

## CONVENTIONS

- **Korean everywhere** — docstrings, comments, CLI help, error messages, README, test names. Code identifiers are English.
- **`from __future__ import annotations`** — every source file.
- **Lazy imports** — heavy ML libs (`torch`, `transformers`, `peft`, `trl`) imported inside functions, never at module level. `__init__.py` uses `__getattr__` for deferred loading.
- **Pydantic v2 for config, `@dataclass` for data** — strict separation. Config = `BaseModel` with `model_validator`. Data = plain `@dataclass`.
- **Feature flags** — every optional step checks `config.X.enabled` before executing.
- **Logging** — `from .utils import get_logger; logger = get_logger("module_name")`. Hierarchical `slm_factory.*` namespace. Rich handler.
- **Async bridge** — `run_async()` inside sync Pipeline methods. Teacher-dependent ops use `run_bounded(semaphore, ...)` for concurrency.
- **JSON as IPC** — intermediate outputs saved to JSON files for resume capability.
- **No custom exceptions** — uses built-in `FileNotFoundError`, `RuntimeError`, `ValueError` only.
- **Korean test names** — `def test_정상_문서_반환(self)` pattern with Korean docstrings. Class-based only.
- **Optional deps** — `try/except ImportError` with `None` fallback in `__init__.py` (parsers: DOCX, HWP).

## ANTI-PATTERNS (THIS PROJECT)

- **Zero TODO/FIXME/HACK markers** in codebase — keep it clean.
- **Do NOT import ML libs at module level** — breaks fast CLI startup. Always lazy-import inside functions.
- **Do NOT add `type: ignore`** unless for untyped third-party libs (PyMuPDF, olefile, optional deps). Currently 8 justified suppressions.
- **Silent `except` with `logger.debug()`** exists in `scorer.py:64` and `augmenter.py:59` (JSON parse fallthrough).
- **No CI/CD exists** — no GitHub Actions, no Makefile, no pre-commit hooks, no linter config.
- **Templates are outside package** — `templates/project.yaml` at repo root, not in `src/slm_factory/`. The `importlib.resources` fallback in `config.py` is fragile.

## COMMANDS

```bash
# Install (editable, all optional deps)
pip install -e ".[all]"

# Run tests (no GPU needed — ML libs auto-mocked)
pytest
pytest tests/test_pipeline.py -v
pytest tests/test_cli.py::TestInit -v

# CLI
slm-factory init <name>
slm-factory tool wizard --config project.yaml
slm-factory run --config project.yaml
slm-factory check --config project.yaml
```

## NOTES

- **`cli.py` is 2202 lines** — all 19 commands in one file (incl. `tool compare-data`). Wizard alone is ~400 lines. Known maintenance burden.
- **`config.py` has 30 Pydantic models** — single file, 605 lines. 22 sub-configs composed into `SLMConfig`.
- **Pipeline is stateless** — no mutable state between steps. Resume logic lives in CLI (detects existing output files), not Pipeline.
- **`QAPair` changes ripple to 19+ modules** — treat as a stable interface.
- **`run_async()` in Pipeline** — cannot use Pipeline from an existing event loop (unless `nest_asyncio` installed).
- **Multi-GPU**: `DeviceInfo.gpu_count` detects via `torch.cuda.device_count()`. DDP auto-configured in training args.
- **MPS (Apple Silicon)**: no BitsAndBytes quantization, no bfloat16, no gradient checkpointing. Auto-detected by `device.py`.
- **Ollama required** for Teacher LLM (default backend). Supports NDJSON streaming (`stream=True`). Health check via `teacher.health_check()`.
- **`trust_remote_code=True`** in trainer — security warning logged when used.
