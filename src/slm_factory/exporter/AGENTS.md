# exporter/

Model export subsystem. Three strategies: HuggingFace merge, Ollama deployment, GGUF quantization.

## STRUCTURE

```
exporter/
├── __init__.py         # Re-exports: HFExporter, OllamaExporter, GGUFExporter
├── hf_export.py        # HFExporter — LoRA merge + safetensors save
├── ollama_export.py    # OllamaExporter — Modelfile generation + ollama create
└── gguf_export.py      # GGUFExporter — llama.cpp convert_hf_to_gguf.py
```

## HOW IT WORKS

1. `HFExporter.export(adapter_path)` → loads base model + LoRA adapter → merges → saves to `output/merged_model/`.
2. `OllamaExporter.export(model_dir)` → generates `Modelfile` with system prompt + parameters → runs `ollama create` via subprocess.
3. `GGUFExporter.export(model_dir)` → calls llama.cpp's `convert_hf_to_gguf.py` via subprocess → quantizes.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Change merge behavior | `hf_export.py` | Uses `device.py` for dtype/device_map |
| Change Ollama Modelfile | `ollama_export.py` | Template uses `OllamaExportConfig` |
| Change quantization type | `gguf_export.py` | Valid types in `GGUFExportConfig` validator |
| Add new export format | New file + `__init__.py` | Wire in `pipeline.step_export()` |

## CONVENTIONS

- Heavy ML imports (`torch`, `transformers`, `peft`) are lazy — inside methods, not at module level.
- Subprocess calls (`ollama create`, `convert_hf_to_gguf.py`) use `subprocess.run()` with error checking.
- `DeviceInfo` from `device.py` determines model loading dtype and device_map (incl. multi-GPU).
- `RuntimeError` raised for missing models, failed subprocess calls, missing llama.cpp.

## ANTI-PATTERNS

- Do NOT import `torch` at module level — exporter may be imported for config validation without GPU.
- `ollama create` and `ollama rm` are destructive — the `EvolveHistory` class manages version cleanup.
