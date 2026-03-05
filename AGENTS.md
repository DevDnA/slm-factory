# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-05
**Commit:** cfb8124
**Branch:** main

## OVERVIEW

Teacher-Student knowledge distillation CLI framework for building domain-specific Small Language Models (SLMs) from documents. Korean-authored. Python 3.11+, Typer CLI, Pydantic v2 config, PyTorch/HuggingFace ML stack, Ollama deployment.

## STRUCTURE

```
slm-factory/
├── src/slm_factory/          # Source (src layout, setuptools)
│   ├── cli.py                # ALL CLI commands — monolith (1899 lines)
│   ├── pipeline.py           # Orchestrator — 12 step_* methods
│   ├── config.py             # 28 Pydantic v2 models for YAML config
│   ├── models.py             # Shared dataclasses: QAPair, ParsedDocument, etc.
│   ├── device.py             # CUDA/MPS/CPU auto-detection
│   ├── utils.py              # Logging (Rich), async helpers, hash utils
│   ├── analyzer.py           # QA data statistics
│   ├── augmenter.py          # Paraphrase augmentation via Teacher LLM
│   ├── comparator.py         # Base vs fine-tuned model comparison
│   ├── converter.py          # QA → chat-template JSONL
│   ├── evaluator.py          # BLEU/ROUGE evaluation
│   ├── evolve_history.py     # Versioned model evolution tracking
│   ├── incremental.py        # Document change detection (hash-based)
│   ├── scorer.py             # LLM-based QA quality scoring
│   ├── parsers/              # → see parsers/AGENTS.md
│   ├── teacher/              # → see teacher/AGENTS.md
│   ├── exporter/             # → see exporter/AGENTS.md
│   ├── trainer/              # → see trainer/AGENTS.md
│   ├── validator/            # → see validator/AGENTS.md
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
| Add pipeline step | `pipeline.py` + `config.py` | New `step_*` method + config model + `SLMConfig` field |
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
| `SLMConfig` | class | `config.py:430` | Root Pydantic v2 config (20 sub-configs) |
| `load_config` | func | `config.py:474` | YAML → validated `SLMConfig` |
| `app` | Typer | `cli.py:30` | CLI entry point (`slm-factory` command) |
| `QAPair` | dataclass | `models.py:29` | Shared data model — system's lingua franca |
| `ParsedDocument` | dataclass | `models.py:9` | Document parser output |
| `BaseParser` | ABC | `parsers/base.py:48` | Parser interface — `parse(path) → ParsedDocument` |
| `ParserRegistry` | class | `parsers/base.py:64` | Registration + batch parsing with Rich progress |
| `BaseTeacher` | ABC | `teacher/base.py:8` | LLM interface — `generate()` / `agenerate()` |
| `create_teacher` | func | `teacher/__init__.py:22` | Factory: config.backend → teacher instance |
| `DeviceInfo` | dataclass | `device.py:19` | Frozen hardware detection result |
| `detect_device` | func | `device.py:86` | CUDA → MPS → CPU priority detection |
| `EvolveHistory` | class | `evolve_history.py` | Version tracking + quality gate + cleanup |
| `IncrementalTracker` | class | `incremental.py` | Document hash-based change detection |

## CONVENTIONS

- **Korean everywhere** — docstrings, comments, CLI help, error messages, README, test names. Code identifiers are English.
- **`from __future__ import annotations`** — every source file.
- **Lazy imports** — heavy ML libs (`torch`, `transformers`, `peft`, `trl`) imported inside functions, never at module level. `__init__.py` uses `__getattr__` for deferred loading.
- **Pydantic v2 for config, `@dataclass` for data** — strict separation. Config = `BaseModel` with `model_validator`. Data = plain `@dataclass`.
- **Feature flags** — every optional step checks `config.X.enabled` before executing.
- **Logging** — `from .utils import get_logger; logger = get_logger("module_name")`. Hierarchical `slm_factory.*` namespace. Rich handler.
- **Async bridge** — `asyncio.run()` inside sync Pipeline methods. Teacher-dependent ops use `run_bounded(semaphore, ...)` for concurrency.
- **JSON as IPC** — intermediate outputs saved to JSON files for resume capability.
- **No custom exceptions** — uses built-in `FileNotFoundError`, `RuntimeError`, `ValueError` only.
- **Korean test names** — `def test_정상_문서_반환(self)` pattern with Korean docstrings.

## ANTI-PATTERNS (THIS PROJECT)

- **Zero TODO/FIXME/HACK markers** in codebase — keep it clean.
- **Do NOT import ML libs at module level** — breaks fast CLI startup. Always lazy-import inside functions.
- **Do NOT add `type: ignore`** unless for untyped third-party libs (PyMuPDF, optional deps). Currently 6 justified suppressions.
- **Silent `except: pass`** exists in `scorer.py:64` and `augmenter.py:59` (JSON parse fallthrough) — add `logger.debug()` if touching these.
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

- **`cli.py` is 1899 lines** — all 18 commands in one file. Wizard alone is ~400 lines. Known maintenance burden.
- **`config.py` has 28 Pydantic models** — single file, 545 lines. 20 sub-configs composed into `SLMConfig`.
- **Pipeline is stateless** — no mutable state between steps. Resume logic lives in CLI, not Pipeline.
- **`QAPair` changes ripple to 19+ modules** — treat as a stable interface.
- **`asyncio.run()` in Pipeline** — cannot use Pipeline from an existing event loop.
- **MPS (Apple Silicon)**: no BitsAndBytes quantization, no bfloat16, no gradient checkpointing. Auto-detected by `device.py`.
- **Ollama required** for Teacher LLM (default backend). Health check via `teacher.health_check()`.
