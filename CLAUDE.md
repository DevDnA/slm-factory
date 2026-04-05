# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slm-factory is a Teacher-Student Knowledge Distillation framework for building domain-specific Small Language Models (SLMs). It uses a large Teacher LLM (9B via Ollama) to generate QA training data from domain documents, then fine-tunes a small Student model (1B) with LoRA. RAG handles factual knowledge (WHAT); fine-tuning teaches response style (HOW).

Two main usage patterns:
- `slf rag` — instant RAG + Teacher model chat (30 seconds setup)
- `slf tune` — fine-tune Student + RAG + chat (30 minutes)

## Commands

```bash
# Setup
./setup.sh                          # One-click: installs uv, deps, Ollama, Teacher model

# Run tests
uv run pytest                       # All tests
uv run pytest tests/test_cli.py -v  # Single file
uv run pytest -k "test_name"        # Single test by name

# CLI (use slf wrapper or uv run)
slf init my-project                 # Create project from template
slf rag                             # Start RAG + chat
slf tune                            # Full pipeline: parse → generate → train → RAG → chat
slf train                           # Training pipeline only (no chat)
slf check                           # Validate docs/config without processing
slf export                          # Export model to Ollama/HuggingFace
slf eval run                        # Evaluate model performance
slf status                          # Show project progress
slf clean                           # Clean artifacts

# Dependencies
uv sync --extra all                 # Install all optional deps
uv sync --extra dev                 # Dev deps only (pytest)
```

The `slf` wrapper runs `uv run --project <repo-dir> slm-factory [args]` — no venv activation needed.

## Architecture

### Pipeline (13 steps, each outputs JSON for resumability)

Parse → Generate QA → Validate → Score → Augment → Analyze → Convert → Train → Export → Eval → Refine → AutoRAG Export → RAG Index

Orchestrated by `Pipeline` class in `pipeline.py`, CLI in `cli.py` (Typer).

### Module Layout (`src/slm_factory/`)

| Module | Role |
|---|---|
| `config.py` | Pydantic v2 config model (28 nested sections), loaded from `project.yaml` |
| `models.py` | Data models: ParsedDocument, QAPair, EvalResult, CompareResult |
| `pipeline.py` | Step orchestration, checkpoint resume |
| `cli.py` | Typer CLI (~1700 lines), all user-facing commands |
| `parsers/` | Registry-pattern parsers for 12 formats (PDF, HWPX, DOCX, HWP, HTML, TXT, MD, PPT, PPTX, XLSX, XLS, DOC) |
| `teacher/` | LLM abstraction — `BaseTeacher` ABC with `OllamaTeacher` and `OpenAICompatTeacher` |
| `teacher/qa_generator.py` | Document → QA pair generation via Teacher LLM |
| `validator/` | QA filtering: `rules.py` (regex rejection), `similarity.py` (semantic groundedness) |
| `scorer.py` | Teacher LLM quality scoring (1-5 scale) |
| `augmenter.py` | Question paraphrasing for data augmentation |
| `trainer/lora_trainer.py` | LoRA fine-tuning via HuggingFace TRL SFTTrainer |
| `exporter/` | Export: HuggingFace merge, Ollama Modelfile, AutoRAG parquet |
| `rag/indexer.py` | Qdrant vector indexing, hybrid search (vector + BM25), cross-encoder reranking |
| `rag/server.py` | FastAPI RAG server (`/v1/query`, `/chat`) with Ollama integration |
| `calibration.py` | Auto-calibrate chunk size, epochs, LR based on data statistics |
| `device.py` | GPU/MPS/CPU detection |
| `evaluator.py` | BLEU/ROUGE auto-evaluation |
| `incremental.py` | Document hash-based change tracking for incremental processing |

### Key Design Patterns

- **Registry pattern** for parsers (extensible via decorator)
- **Factory pattern** — `create_teacher()` abstracts LLM backend selection
- **Lazy imports** — heavy ML libraries (torch, transformers) loaded only when needed
- **TYPE_CHECKING imports** to minimize startup time
- **Async concurrency** — parallel LLM/embedding calls via asyncio
- **Silent failure isolation** — skip bad files, don't halt entire pipeline

### Configuration

Projects are configured via `project.yaml` (template in `templates/`). Pydantic v2 validates all config with detailed error messages.

## Conventions

- **Language**: Korean docstrings/docs, English code identifiers
- Python 3.11+ with `from __future__ import annotations`
- No configured linter/formatter (no ruff/black/mypy config)
- Tests mock all ML libraries (torch, transformers, etc.) via `conftest.py` fixtures for fast execution
- Test fixtures: `make_config()`, `make_qa_pair()`, `make_parsed_doc()` factories in `tests/conftest.py`
