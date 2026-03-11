# exporter/

Model export subsystem. Four strategies: HuggingFace merge, Ollama deployment, GGUF quantization, AutoRAG data export.

## STRUCTURE

```
exporter/
├── __init__.py         # Re-exports: HFExporter, OllamaExporter, GGUFExporter, AutoRAGExporter
├── hf_export.py        # HFExporter — LoRA merge + safetensors save
├── ollama_export.py    # OllamaExporter — Modelfile generation + ollama create
├── gguf_export.py      # GGUFExporter — llama.cpp convert_hf_to_gguf.py
└── autorag_export.py   # AutoRAGExporter — docs+QA → corpus.parquet + qa.parquet
```

## HOW IT WORKS

1. `HFExporter.export(adapter_path)` → loads base model + LoRA adapter → merges → saves to `output/merged_model/`.
2. `OllamaExporter.export(model_dir)` → generates `Modelfile` with system prompt + parameters → runs `ollama create` via subprocess.
3. `GGUFExporter.export(model_dir)` → 2-stage pipeline: Stage 1 calls `convert_hf_to_gguf.py` (HF→f16 GGUF), Stage 2 calls `llama-quantize` for non-convert types (q4_k_m, etc.). Types in `_CONVERT_OUTTYPES` (f32, f16, bf16, q8_0, etc.) skip Stage 2.
4. `AutoRAGExporter.export(docs, pairs)` → chunks documents at paragraph boundaries → generates deterministic UUIDs → writes `corpus.parquet` + `qa.parquet` to `output/autorag/`.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Change merge behavior | `hf_export.py` | Uses `device.py` for dtype/device_map |
| Change Ollama Modelfile | `ollama_export.py` | Template uses `OllamaExportConfig` |
| Change quantization type | `gguf_export.py` | `_CONVERT_OUTTYPES` for 1-stage, others use 2-stage via `llama-quantize` |
| Change AutoRAG chunking | `autorag_export.py:_chunk_for_retrieval()` | Paragraph-boundary splitting |
| Change AutoRAG schema | `autorag_export.py` | corpus.parquet (doc_id, contents, metadata) + qa.parquet (qid, query, retrieval_gt, generation_gt) |
| Add new export format | New file + `__init__.py` | Wire in `pipeline.step_export()` |

## CONVENTIONS

- Heavy ML imports (`torch`, `transformers`, `peft`) are lazy — inside methods, not at module level.
- Subprocess calls (`ollama create`, `convert_hf_to_gguf.py`) use `subprocess.run()` with error checking.
- `DeviceInfo` from `device.py` determines model loading dtype and device_map (incl. multi-GPU).
- `RuntimeError` raised for missing models, failed subprocess calls, missing llama.cpp.
- `AutoRAGExporter` uses deterministic UUIDs via `uuid.uuid5()` — same input always generates same doc_id for idempotent re-runs.
- Requires `pandas` and `pyarrow` at runtime — lazy imports inside methods.

## ANTI-PATTERNS

- Do NOT import `torch` at module level — exporter may be imported for config validation without GPU.
- `ollama create` and `ollama rm` are destructive — the `EvolveHistory` class manages version cleanup.
- Do NOT add quantization types to `_CONVERT_OUTTYPES` unless `convert_hf_to_gguf.py` natively supports them — others must go through `llama-quantize` (Stage 2).
- Do NOT change `_NAMESPACE` UUID in `autorag_export.py` — breaks existing corpus/qa ID linkage across re-runs.
