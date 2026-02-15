"""Ollama 내보내기 — Modelfile을 생성하고 선택적으로 로컬 Ollama로 가져옵니다.

병합된 HuggingFace 모델에서 Ollama 호환 Modelfile을 생성하여
`ollama create`를 통한 원클릭 배포를 가능하게 합니다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("exporter.ollama_export")


class OllamaExporter:
    """Ollama Modelfile을 생성하고 선택적으로 모델을 가져옵니다.
    
    인자:
        config: 전체 SLMConfig
    """
    
    def __init__(self, config: SLMConfig):
        self.config = config
        self.ollama_config = config.export.ollama
        self.model_name = config.export.ollama.model_name
        self.system_prompt = config.export.ollama.system_prompt
        self.parameters = config.export.ollama.parameters
    
    def generate_modelfile(
        self,
        model_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Ollama Modelfile을 생성합니다.
        
        인자:
            model_dir: HuggingFace 모델 디렉토리의 경로
            output_path: Modelfile의 출력 경로 (기본값: model_dir / "Modelfile")
        
        반환값:
            생성된 Modelfile의 경로
        """
        model_dir = Path(model_dir)
        if output_path is None:
            output_path = model_dir / "Modelfile"
        output_path = Path(output_path)
        
        logger.info(f"Modelfile 생성 중: {output_path}")
        
        # Modelfile 내용 구성
        lines = [
            f"FROM {model_dir}",
            "",
            'SYSTEM """',
            self.system_prompt,
            '"""',
            "",
        ]
        
        # 매개변수 추가
        for param_name, param_value in self.parameters.items():
            lines.append(f"PARAMETER {param_name} {param_value}")
        
        content = "\n".join(lines)
        
        # Modelfile 작성
        output_path.write_text(content, encoding="utf-8")
        
        logger.info(f"✓ Modelfile 생성됨: {output_path}")
        
        return output_path
    
    def create_model(self, modelfile_path: str | Path) -> bool:
        """Modelfile에서 Ollama 모델을 생성합니다.
        
        `ollama create` 명령을 실행하여 모델을 가져옵니다.
        
        인자:
            modelfile_path: Modelfile의 경로
        
        반환값:
            성공하면 True, 그렇지 않으면 False
        """
        from rich.console import Console
        
        console = Console()
        modelfile_path = Path(modelfile_path)
        
        logger.info(f"Ollama 모델 생성 중: {self.model_name}")
        
        try:
            with console.status(f"[bold blue]Ollama 모델 생성 중: {self.model_name}[/bold blue]"):
                result = subprocess.run(
                    ["ollama", "create", self.model_name, "-f", str(modelfile_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5분 타임아웃
                )
            
            if result.returncode == 0:
                logger.info(f"✓ Ollama 모델 생성됨: {self.model_name}")
                if result.stdout:
                    logger.debug(f"표준 출력: {result.stdout}")
                return True
            else:
                logger.warning(f"Ollama 모델 생성 실패 (종료 코드 {result.returncode})")
                if result.stderr:
                    logger.warning(f"표준 오류: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning("ollama 명령을 찾을 수 없음")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("ollama create 명령 타임아웃")
            return False
        except Exception as e:
            logger.warning(f"ollama create 실행 중 오류: {e}")
            return False
    
    def export(
        self,
        model_dir: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        """모델을 Ollama 형식으로 내보냅니다.
        
        Modelfile을 생성하고 선택적으로 모델을 생성하는 주요 진입점입니다.
        
        인자:
            model_dir: HuggingFace 모델 디렉토리의 경로
            output_dir: Modelfile의 출력 디렉토리 (기본값: model_dir)
        
        반환값:
            생성된 Modelfile의 경로
        """
        model_dir = Path(model_dir)
        
        # 출력 경로 결정
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            modelfile_path = output_dir / "Modelfile"
        else:
            modelfile_path = None
        
        # Modelfile 생성
        modelfile_path = self.generate_modelfile(model_dir, modelfile_path)
        
        # Ollama 사용 가능 여부 확인
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ollama_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ollama_available = False
        
        if ollama_available:
            logger.info("Ollama 감지됨, 모델 생성 시도 중...")
            success = self.create_model(modelfile_path)
            if success:
                logger.info(f"✓ 모델 준비 완료! 실행: ollama run {self.model_name}")
            else:
                logger.info(f"수동으로 가져오려면 실행: ollama create {self.model_name} -f {modelfile_path}")
        else:
            logger.info("Ollama를 감지하지 못했습니다. 수동으로 가져오려면:")
            logger.info(f"  1. https://ollama.ai에서 Ollama 설치")
            logger.info(f"  2. 실행: ollama create {self.model_name} -f {modelfile_path}")
        
        return modelfile_path
