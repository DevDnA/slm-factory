"""HuggingFace TRL의 SFTTrainer를 이용한 LoRA(Low-Rank Adaptation) 미세 조정.

학생 모델에 LoRA 어댑터를 적용하고, 조기 종료(Early Stopping)와
코사인 학습률 스케줄링으로 학습한 후 어댑터 가중치를 저장합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import SLMConfig

from transformers import TrainerCallback

from ..utils import get_logger

logger = get_logger("trainer.lora_trainer")


class _RichProgressCallback(TrainerCallback):
    """학습 진행 상황을 로거를 통해 표시하는 콜백입니다."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch is not None:
            logger.info("Epoch %d/%d 시작", int(state.epoch) + 1, int(args.num_train_epochs))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            parts = []
            if "loss" in logs:
                parts.append(f"loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                parts.append(f"eval_loss={logs['eval_loss']:.4f}")
            if "learning_rate" in logs:
                parts.append(f"lr={logs['learning_rate']:.2e}")
            if parts:
                logger.info("step %d  %s", state.global_step, "  ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        epoch_str = f"{state.epoch:.1f}" if state.epoch else "?"
        logger.info("학습 완료 — 총 %d 스텝, 최종 epoch %s", state.global_step, epoch_str)


class LoRATrainer:
    """LoRA(Low-Rank Adaptation) 미세 조정 오케스트레이터.

    HuggingFace TRL의 SFTTrainer를 적절한 LoRA 설정, 조기 종료,
    그리고 설정의 모든 학습 하이퍼파라미터로 감싸서 제공합니다.
    """

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.student_config = config.student
        self.training_config = config.training
        self.lora_config = config.training.lora
        self.output_dir = Path(config.paths.output) / "checkpoints"
        self._device_info: Any = None

    def _load_model(self) -> tuple[Any, Any]:
        """학생 모델과 토크나이저를 로드합니다.

        디바이스를 자동 감지하여 CUDA/MPS/CPU에 맞는 설정을 적용합니다.
        CUDA에서만 BitsAndBytes 4비트 양자화를 지원합니다.

        반환값
        -------
        tuple[model, tokenizer]
            로드된 인과 언어 모델(Causal LM)과 그 토크나이저.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        from ..device import detect_device

        device = detect_device()
        self._device_info = device

        model_name = self.student_config.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        tokenizer.padding_side = "right"

        # 양자화 설정 (CUDA + bitsandbytes 사용 가능 시에만)
        quantization_config = None
        quant_cfg = self.training_config.quantization
        if quant_cfg.enabled and device.quantization_available:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=device.torch_dtype,
            )
            logger.info("Quantization enabled: %d-bit NF4", quant_cfg.bits)
        elif quant_cfg.enabled and not device.quantization_available:
            if device.type == "mps":
                logger.warning(
                    "Apple Silicon에서는 BitsAndBytes 양자화를 사용할 수 없습니다. "
                    "Unified Memory를 활용하여 양자화 없이 진행합니다."
                )
            else:
                logger.warning(
                    "BitsAndBytes가 설치되지 않았습니다. 양자화 없이 진행합니다. "
                    "설치: pip install bitsandbytes"
                )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device.device_map,
                torch_dtype=device.torch_dtype,
                trust_remote_code=True,
            )
        except OSError as e:
            if "does not appear to have" in str(e) or "not found" in str(e).lower():
                raise RuntimeError(
                    f"학생 모델 '{model_name}'을(를) 찾을 수 없습니다. "
                    f"모델명이 정확한지, 인터넷 연결이 되어 있는지 확인하세요. "
                    f"HuggingFace 모델 검색: https://huggingface.co/models?search={model_name}"
                ) from e
            raise
        except RuntimeError as e:
            error_lower = str(e).lower()
            if "cuda" in error_lower:
                raise RuntimeError(
                    "CUDA를 사용할 수 없습니다. GPU 드라이버와 PyTorch CUDA 버전을 확인하세요. "
                    "CPU로 학습하려면 training.bf16을 false로 설정하세요."
                ) from e
            if "mps" in error_lower:
                raise RuntimeError(
                    "MPS 디바이스에서 오류가 발생했습니다. "
                    "PyTorch 버전을 확인하세요 (2.1 이상 권장). "
                    "문제가 지속되면 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 환경변수를 설정하세요."
                ) from e
            raise

        logger.info(
            "Loaded model %s (%.1fM params, device=%s, dtype=%s)",
            model_name,
            sum(p.numel() for p in model.parameters()) / 1e6,
            next(model.parameters()).device,
            device.dtype_name,
        )
        return model, tokenizer

    def _create_lora_config(self) -> Any:
        """학습 설정에서 peft LoraConfig를 구성합니다.

        반환값
        -------
        peft.LoraConfig
            설정된 LoRA 어댑터 설정.
        """
        from peft import LoraConfig, TaskType

        target_modules = self.lora_config.target_modules
        if target_modules == "auto":
            target_modules = None

        lora_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=target_modules,
            use_rslora=self.lora_config.use_rslora,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        logger.info(
            "LoRA config: r=%d, alpha=%d, dropout=%.3f, rslora=%s",
            self.lora_config.r,
            self.lora_config.alpha,
            self.lora_config.dropout,
            self.lora_config.use_rslora,
        )
        return lora_config

    def _create_training_args(self) -> Any:
        """설정에서 HuggingFace TrainingArguments를 구성합니다.

        디바이스에 맞게 bf16/fp16, 옵티마이저 등을 자동 재정의합니다.

        반환값
        -------
        transformers.TrainingArguments
            완전한 학습 인자 명세.
        """
        from transformers import TrainingArguments

        from ..device import get_training_overrides

        tc = self.training_config
        device = getattr(self, "_device_info", None)

        # 디바이스에 맞는 파라미터 재정의
        if device is not None:
            overrides = get_training_overrides(device)
        else:
            overrides = {}

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.batch_size,
            per_device_eval_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            lr_scheduler_type=tc.lr_scheduler,
            warmup_ratio=tc.warmup_ratio,
            optim=overrides.get("optim", tc.optimizer),
            bf16=overrides.get("bf16", tc.bf16),
            fp16=overrides.get("fp16", False),
            logging_steps=10,
            eval_strategy=tc.save_strategy,
            save_strategy=tc.save_strategy,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=42,
            remove_unused_columns=False,
        )

        precision = "bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "fp32")
        logger.info(
            "Training: %d epochs, batch=%d, grad_accum=%d, lr=%.2e, scheduler=%s, "
            "precision=%s, optim=%s",
            tc.num_epochs,
            tc.batch_size,
            tc.gradient_accumulation_steps,
            tc.learning_rate,
            tc.lr_scheduler,
            precision,
            training_args.optim,
        )
        return training_args

    def _create_callbacks(self) -> list[Any]:
        """학습 콜백(조기 종료 등)을 구성합니다.

        반환값
        -------
        list
            HuggingFace 콜백 인스턴스 목록.
        """
        from transformers import EarlyStoppingCallback

        callbacks: list[Any] = []

        es_cfg = self.training_config.early_stopping
        if es_cfg.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=es_cfg.patience,
                    early_stopping_threshold=es_cfg.threshold,
                )
            )
            logger.info(
                "Early stopping enabled: patience=%d, threshold=%.4f",
                es_cfg.patience,
                es_cfg.threshold,
            )

        callbacks.append(_RichProgressCallback())

        return callbacks

    def train(self, dataset_dict: Any) -> Path:
        """제공된 데이터셋에서 LoRA 미세 조정을 실행합니다.

        매개변수
        ----------
        dataset_dict:
            ``"train"``과 ``"eval"`` 분할을 포함하는 ``DatasetDict``,
            각각 ``"text"`` 열을 포함합니다.

        반환값
        -------
        Path
            저장된 어댑터 디렉토리의 경로.
        """
        from trl import SFTTrainer
        from peft import get_peft_model

        # 모델과 토크나이저 로드
        model, tokenizer = self._load_model()

        # LoRA 어댑터 적용
        lora_config = self._create_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 학습 인자와 콜백 구성
        training_args = self._create_training_args()
        callbacks = self._create_callbacks()

        # 트레이너 생성
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["eval"],
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        # 학습 실행
        device = getattr(self, "_device_info", None)
        device_type = device.type if device else "unknown"
        logger.info("LoRA 미세 조정 시작 (디바이스: %s)...", device_type)
        try:
            trainer.train()
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg or "mps" in error_msg:
                logger.error("GPU 메모리 부족 — 다음을 시도하세요:")
                logger.error("  1. batch_size를 줄이세요 (현재: %d)", self.training_config.batch_size)
                logger.error("  2. gradient_accumulation_steps를 늘리세요 (현재: %d)", self.training_config.gradient_accumulation_steps)
                if device_type == "cuda":
                    logger.error("  3. quantization.enabled를 true로 설정하세요")
                elif device_type == "mps":
                    logger.error("  3. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 환경변수를 설정하세요")
                logger.error("  4. LoRA r 값을 줄이세요 (현재: %d)", self.lora_config.r)
                raise RuntimeError(
                    "GPU 메모리가 부족합니다. project.yaml에서 batch_size를 줄이거나 "
                    "gradient_accumulation_steps를 늘려보세요."
                ) from e
            raise

        # 어댑터와 토크나이저 저장
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        logger.info("학습 완료 — 어댑터가 %s에 저장됨", adapter_dir)
        return adapter_dir


_data_logger = get_logger("trainer.data_loader")


class DataLoader:
    """JSONL 파일에서 학습 데이터를 로드하고 분할합니다.

    JSONL 형식: 한 줄에 하나의 {"text": "..."}, "text"는
    채팅 템플릿 형식의 학습 문자열을 포함합니다.
    """

    def __init__(self, train_split: float = 0.9) -> None:
        self.train_split = train_split

    def load_jsonl(self, path: str | Path) -> "datasets.Dataset":
        """JSONL 파일을 HuggingFace Dataset으로 로드합니다.

        매개변수
        ----------
        path:
            한 줄에 하나의 ``{"text": "..."}``를 포함하는 ``.jsonl`` 파일의 경로.

        반환값
        -------
        datasets.Dataset
            로드된 데이터셋.
        """
        from datasets import load_dataset

        path = Path(path)
        ds = load_dataset("json", data_files=str(path), split="train")
        _data_logger.info("Loaded %d examples from %s", len(ds), path.name)
        return ds

    def split(self, dataset: "datasets.Dataset") -> "datasets.DatasetDict":
        """데이터셋을 학습 및 평가 세트로 분할합니다.

        매개변수
        ----------
        dataset:
            분할할 HuggingFace Dataset.

        반환값
        -------
        datasets.DatasetDict
            ``"train"``과 ``"eval"`` 키를 포함하는 딕셔너리.
        """
        from datasets import DatasetDict

        split = dataset.train_test_split(test_size=1 - self.train_split, seed=42)
        ds_dict = DatasetDict({
            "train": split["train"],
            "eval": split["test"],
        })
        _data_logger.info(
            "Split: %d train / %d eval (%.0f%% train)",
            len(ds_dict["train"]),
            len(ds_dict["eval"]),
            self.train_split * 100,
        )
        return ds_dict

    def load_and_split(self, path: str | Path) -> "datasets.DatasetDict":
        """JSONL 파일을 로드하고 학습/평가 세트로 분할합니다.

        :meth:`load_jsonl`과 :meth:`split`을 연결하는 편의 메서드입니다.

        매개변수
        ----------
        path:
            ``.jsonl`` 파일의 경로.

        반환값
        -------
        datasets.DatasetDict
            ``"train"``과 ``"eval"`` 키를 포함하는 딕셔너리.
        """
        dataset = self.load_jsonl(path)
        return self.split(dataset)
