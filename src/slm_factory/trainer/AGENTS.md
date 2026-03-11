# trainer/

LoRA fine-tuning subsystem. DataLoader for dataset preparation, LoRATrainer for PEFT-based training with device-aware configuration. Multi-GPU DDP support.

## STRUCTURE

```
trainer/
├── __init__.py        # Re-exports: DataLoader, LoRATrainer
└── lora_trainer.py    # DataLoader + LoRATrainer + _RichProgressCallback
```

## HOW IT WORKS

1. `DataLoader.load_and_split(path)` reads chat-template JSONL, splits into train/eval `DatasetDict`.
2. `LoRATrainer.train(dataset_dict)` loads base model + tokenizer, applies LoRA adapter, runs SFTTrainer.
3. Device detection (`device.py`) auto-selects dtype, optimizer, quantization, gradient checkpointing.
4. Multi-GPU: `DeviceInfo.gpu_count > 1` → auto-configures DDP in training args.
5. `_RichProgressCallback` logs epoch/loss/lr to Rich console during training.
6. Early stopping via HuggingFace `EarlyStoppingCallback` when `config.training.early_stopping.enabled`.
7. `trust_remote_code=True` support — security warning logged when enabled.
8. `_MPSCacheCleanupCallback` clears MPS tensor cache (`torch.mps.empty_cache()`) after each training step to prevent memory accumulation and swap storm on Apple Silicon.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change model loading | `lora_trainer.py:_load_model()` | Lazy imports torch/transformers here |
| Change LoRA config | `lora_trainer.py:_create_lora_config()` | Maps `config.training.lora` to peft `LoraConfig` |
| Change training args | `lora_trainer.py:_create_training_args()` | Merges user config + device overrides |
| Change callbacks | `lora_trainer.py:_create_callbacks()` | Rich progress + optional early stopping |
| Change dataset loading | `lora_trainer.py:DataLoader` | JSONL → HuggingFace `Dataset` → train/eval split |
| Modify training loop | `lora_trainer.py:LoRATrainer.train()` | Creates SFTTrainer, handles device placement |
| Change MPS memory handling | `lora_trainer.py:_MPSCacheCleanupCallback` | Auto-added when device is MPS |

## CONVENTIONS

- All ML imports (`torch`, `transformers`, `peft`, `trl`, `datasets`) are **lazy** — inside methods, never at module level.
- `_RichProgressCallback` subclasses `transformers.TrainerCallback` (imported at module level — sole exception for inheritance).
- Device-aware: `detect_device()` determines dtype, optimizer, quantization, device_map automatically.
- Adapter saved to `{output_dir}/lora_adapter/` — Pipeline's `step_export()` reads from here.
- Error handling wraps common failures (OOM, missing model) with Korean log messages and re-raises.
- `_MPSCacheCleanupCallback` auto-injected in `_create_callbacks()` when `device.type == "mps"` — do NOT add manually.

## ANTI-PATTERNS

- Do NOT import `torch`, `transformers`, `peft`, `trl` at module level — breaks fast CLI startup.
- Do NOT hardcode device/dtype — always use `detect_device()` + `get_training_overrides()`.
- Do NOT bypass `SFTTrainer` — it handles chat template formatting and data collation.
