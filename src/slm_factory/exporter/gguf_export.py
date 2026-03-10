"""GGUF 내보내기 — llama.cpp를 사용하여 HuggingFace 모델을 GGUF 형식으로 변환합니다.

병합된 HuggingFace 모델을 llama.cpp의 convert_hf_to_gguf.py 스크립트로
양자화하여 GGUF 형식의 추론 최적화 모델을 생성합니다.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("exporter.gguf_export")


class GGUFExporter:
    """HuggingFace 모델을 GGUF 형식으로 변환합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.quantization_type = config.gguf_export.quantization_type
        self.llama_cpp_path = config.gguf_export.llama_cpp_path
        self.model_name = config.project.name

    _CONVERT_OUTTYPES = frozenset(
        {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
    )

    def export(self, model_dir: Path) -> Path:
        """모델 디렉토리를 GGUF 형식으로 변환합니다."""
        from rich.console import Console

        console = Console()
        model_dir = Path(model_dir)

        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"모델 디렉토리를 찾을 수 없습니다: {model_dir}"
            )

        llama_cpp_dir = self._find_llama_cpp()
        convert_script = self._find_convert_script(llama_cpp_dir)

        needs_quantize = self.quantization_type not in self._CONVERT_OUTTYPES
        convert_outtype = "f16" if needs_quantize else self.quantization_type
        output_file = model_dir / f"{self.model_name}-{self.quantization_type}.gguf"

        logger.info(
            f"GGUF 변환 시작: {model_dir} → {output_file} "
            f"(양자화: {self.quantization_type})"
        )

        try:
            # 1단계: HF → GGUF 변환
            f16_file = model_dir / f"{self.model_name}-f16.gguf" if needs_quantize else output_file
            with console.status(
                f"[bold blue]HF → GGUF 변환 중 ({convert_outtype})[/bold blue]"
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(convert_script),
                        str(model_dir),
                        "--outtype",
                        convert_outtype,
                        "--outfile",
                        str(f16_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )

            if result.returncode != 0:
                raise RuntimeError(
                    f"GGUF 변환에 실패했습니다 (종료 코드 {result.returncode}). "
                    f"표준 오류: {result.stderr}"
                )

            if result.stdout:
                logger.debug("변환 출력: %s", result.stdout)

            # 2단계: llama-quantize로 양자화 (q4_k_m 등)
            if needs_quantize:
                quantize_bin = shutil.which("llama-quantize")
                if not quantize_bin:
                    raise FileNotFoundError(
                        "llama-quantize를 찾을 수 없습니다. "
                        "llama.cpp를 설치하세요."
                    )

                with console.status(
                    f"[bold blue]양자화 중: {self.quantization_type}[/bold blue]"
                ):
                    q_result = subprocess.run(
                        [quantize_bin, str(f16_file), str(output_file), self.quantization_type],
                        capture_output=True,
                        text=True,
                        timeout=1800,
                    )

                if q_result.returncode != 0:
                    raise RuntimeError(
                        f"양자화에 실패했습니다 (종료 코드 {q_result.returncode}). "
                        f"표준 오류: {q_result.stderr}"
                    )

                if q_result.stdout:
                    logger.debug("양자화 출력: %s", q_result.stdout)

                # f16 중간 파일 정리
                if f16_file.exists() and f16_file != output_file:
                    f16_file.unlink()
                    logger.debug("중간 파일 삭제: %s", f16_file)

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "GGUF 변환이 타임아웃되었습니다. "
                "모델 크기가 너무 크거나 시스템 리소스가 부족할 수 있습니다."
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"변환 스크립트를 실행할 수 없습니다: {e}"
            ) from e

        if not output_file.exists():
            gguf_files = list(model_dir.glob("*.gguf"))
            if gguf_files:
                output_file = gguf_files[0]
            else:
                raise RuntimeError(
                    "GGUF 파일이 생성되지 않았습니다. "
                    "llama.cpp 변환 로그를 확인하세요."
                )

        logger.info("✓ GGUF 파일 생성됨: %s", output_file)

        return output_file

    def _find_llama_cpp(self) -> Path:
        """llama.cpp 디렉토리를 탐색합니다."""
        # 1. 설정에서 지정된 경로 확인
        if self.llama_cpp_path:
            configured_path = Path(self.llama_cpp_path)
            if configured_path.is_dir():
                logger.debug("설정된 llama.cpp 경로 사용: %s", configured_path)
                return configured_path
            raise FileNotFoundError(
                f"설정된 llama.cpp 경로를 찾을 수 없습니다: {configured_path}"
            )

        # 2. 일반적인 위치 검색
        common_paths = [
            Path.home() / "llama.cpp",
            Path.home() / "projects" / "llama.cpp",
            Path("/opt/llama.cpp"),
            Path("/usr/local/share/llama.cpp"),
            Path.cwd() / "llama.cpp",
        ]

        for candidate in common_paths:
            if candidate.is_dir():
                logger.debug("llama.cpp 발견: %s", candidate)
                return candidate

        # 3. PATH에서 검색
        convert_in_path = shutil.which("convert_hf_to_gguf.py")
        if convert_in_path:
            return Path(convert_in_path).parent

        raise FileNotFoundError(
            "llama.cpp를 찾을 수 없습니다. "
            "gguf_export.llama_cpp_path를 설정하거나 "
            "llama.cpp를 설치하세요. "
            "(https://github.com/ggerganov/llama.cpp)"
        )

    def _find_convert_script(self, llama_cpp_dir: Path) -> Path:
        """llama.cpp 디렉토리에서 convert_hf_to_gguf.py를 탐색합니다."""
        # 스크립트가 있을 수 있는 위치
        candidates = [
            llama_cpp_dir / "convert_hf_to_gguf.py",
            llama_cpp_dir / "convert-hf-to-gguf.py",
            llama_cpp_dir / "scripts" / "convert_hf_to_gguf.py",
        ]

        for candidate in candidates:
            if candidate.is_file():
                logger.debug("변환 스크립트 발견: %s", candidate)
                return candidate

        raise FileNotFoundError(
            f"convert_hf_to_gguf.py를 찾을 수 없습니다. "
            f"llama.cpp 디렉토리({llama_cpp_dir})에 변환 스크립트가 "
            f"있는지 확인하세요."
        )
