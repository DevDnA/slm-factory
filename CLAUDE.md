# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

slm-factory is a Teacher-Student Knowledge Distillation framework for building domain-specific Small Language Models (SLMs). It uses a Teacher LLM (default `gemma4:e2b` — Gemma4 Effective 2B — via Ollama; `qwen3.5:9b` recommended for higher multilingual quality) to generate QA training data from domain documents, then fine-tunes a small Student model (1B) with LoRA. RAG handles factual knowledge (WHAT); fine-tuning teaches response style (HOW).

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

## Known Issues & Troubleshooting

### Student Model Selection

- **Gemma-3 (`google/gemma-3-1b-it`) Ollama 호환 문제**: safetensors → GGUF 변환 시 vocab 크기 불일치 및 `model_type: "gemma3_text"` 인식 실패로 빈 응답 또는 깨진 출력 발생. transformers로 직접 추론하면 정상이지만 Ollama에서는 동작하지 않음.
- **권장 Student 모델**: `Qwen/Qwen2.5-1.5B-Instruct` — Ollama GGUF 변환 호환성 우수, 한국어 성능 양호, HF_TOKEN 불필요(공개 모델).

### Ollama Export (`exporter/ollama_export.py`)

- Ollama Modelfile 생성 시 **TEMPLATE 지시어가 누락**되면 `{{ .Prompt }}` (raw passthrough)로 설정되어 chat 형식이 적용되지 않음. 모델별 올바른 chat template 필요.
- Ollama 0.19.0 기준 safetensors → GGUF 내부 변환 시 ARM/neon 아키텍처에서 `Quantization is not supported for ArchType::neon` 경고가 발생하며, 비양자화 F16 GGUF를 생성함. 공식 Ollama 모델(Q4_K_M)과 다른 포맷.

### LoRA Training (`trainer/lora_trainer.py`)

- **Completion-only loss (기본 활성)**: `training.completion_only_loss: true` (기본값)로 assistant 응답 토큰에만 loss를 계산합니다. 학습 시 `{"text": ...}` → `{"prompt": ..., "completion": ...}` 자동 변환 후 `SFTConfig(completion_only_loss=True)`를 사용합니다. 채팅 템플릿에서 assistant 마커를 자동 감지하며, 감지 실패 시 자동으로 비활성화됩니다.
- **소규모 데이터(<100개) 학습 시 주의사항**:
  - `gradient_accumulation_steps`가 학습 데이터 수보다 크면 step 수가 1~2개로 사실상 학습 안 됨 (step = data_count / grad_accum)
  - `label_smoothing_factor > 0.1`은 소규모 데이터에서 학습을 방해할 수 있음 → 0.0 권장
  - MPS(Apple Silicon)에서 `quantization.enabled: true` (4bit)는 수치 불안정 유발 가능 → `false` 권장
  - `learning_rate: auto` 시 calibration.py가 데이터 수 기준으로 결정 (< 100: 5e-5, < 500: 1e-4, 500+: 2e-4)

### 검증된 학습 파라미터 (29개 QA, MPS, Qwen2.5-1.5B)

```yaml
student:
  model: "Qwen/Qwen2.5-1.5B-Instruct"
training:
  lora: { r: 8, alpha: 8, dropout: 0.1 }
  batch_size: 1
  gradient_accumulation_steps: 4    # 26개 데이터 → ~6 steps/epoch
  learning_rate: 3e-5
  num_epochs: 3                     # 총 ~18 steps
  quantization: { enabled: false }  # MPS에서 안정성 확보
  weight_decay: 0.01
  label_smoothing_factor: 0.0
  neftune_noise_alpha: 5.0
rag:
  max_tokens: 512                   # 무한 생성 방지
  request_timeout: 300.0
```
