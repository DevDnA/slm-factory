"""HuggingFace 모델 내보내기 — LoRA 어댑터를 병합하고 전체 모델을 저장합니다.

LoRA 가중치를 기본 모델에 병합하고 배포 또는 추가 변환을 위해
safetensors 형식으로 저장하는 학습 후 단계를 처리합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("exporter.hf_export")


class HFExporter:
    """LoRA 어댑터를 기본 모델에 병합하고 저장합니다.
    
    인자:
        config: 전체 SLMConfig
    """
    
    def __init__(self, config: SLMConfig):
        self.config = config
        self.student_model = config.student.model
        self.merge_lora = config.export.merge_lora
        self.output_format = config.export.output_format
    
    def merge_and_save(
        self,
        adapter_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        """LoRA 어댑터를 기본 모델에 병합하고 저장합니다.
        
        인자:
            adapter_path: LoRA 어댑터 체크포인트의 경로
            output_dir: 출력 디렉토리 (기본값: config.paths.output / "merged_model")
        
        반환값:
            저장된 병합 모델의 경로
        """
        # 지연 임포트
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        adapter_path = Path(adapter_path)
        if output_dir is None:
            output_dir = self.config.paths.output / "merged_model"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"기본 모델 로드 중: {self.student_model}")
        
        # 디바이스 결정
        device_map = "auto" if torch.cuda.is_available() else None
        
        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.student_model,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
        logger.info(f"LoRA 어댑터 로드 중: {adapter_path}")
        
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("LoRA 가중치를 기본 모델에 병합 중...")
        
        # 병합 및 언로드
        model = model.merge_and_unload()
        
        logger.info(f"병합된 모델을 저장 중: {output_dir}")
        
        # 병합된 모델 저장
        model.save_pretrained(
            output_dir,
            safe_serialization=True,  # safetensors 형식 사용
        )
        
        # 토크나이저 저장
        logger.info("토크나이저 저장 중...")
        tokenizer = AutoTokenizer.from_pretrained(self.student_model)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✓ 병합된 모델이 저장됨: {output_dir}")
        
        return output_dir
    
    def save_adapter_only(
        self,
        adapter_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        """병합 없이 LoRA 어댑터를 저장합니다.
        
        모델 크기를 작게 유지하고 유연성을 유지하는 데 유용합니다.
        
        인자:
            adapter_path: LoRA 어댑터 체크포인트의 경로
            output_dir: 출력 디렉토리 (기본값: config.paths.output / "adapter")
        
        반환값:
            저장된 어댑터의 경로
        """
        import shutil
        
        adapter_path = Path(adapter_path)
        if output_dir is None:
            output_dir = self.config.paths.output / "adapter"
        output_dir = Path(output_dir)
        
        logger.info(f"{adapter_path}에서 {output_dir}로 어댑터 복사 중")
        
        # 어댑터 디렉토리 복사
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(adapter_path, output_dir)
        
        logger.info(f"✓ 어댑터가 저장됨: {output_dir}")
        
        return output_dir
    
    def export(
        self,
        adapter_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        """설정에 따라 모델을 내보냅니다.
        
        병합할지 또는 어댑터만 저장할지 결정하는 주요 진입점.
        
        인자:
            adapter_path: LoRA 어댑터 체크포인트의 경로
            output_dir: 출력 디렉토리 (기본값: 자동 결정)
        
        반환값:
            내보낸 모델/어댑터의 경로
        """
        adapter_path = Path(adapter_path)
        
        if self.merge_lora:
            logger.info("내보내기 모드: LoRA를 기본 모델에 병합")
            return self.merge_and_save(adapter_path, output_dir)
        else:
            logger.info("내보내기 모드: 어댑터만 저장 (병합 없음)")
            return self.save_adapter_only(adapter_path, output_dir)
