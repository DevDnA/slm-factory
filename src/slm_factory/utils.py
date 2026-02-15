"""Rich를 사용한 구조화된 로깅."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """루트 slm-factory 로거를 구성하고 반환합니다."""
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        )
        _configured = True
    return logging.getLogger("slm_factory")


def get_logger(name: str) -> logging.Logger:
    """slm_factory 네임스페이스 아래의 자식 로거를 가져옵니다."""
    return logging.getLogger(f"slm_factory.{name}")


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """파일의 해시값을 계산합니다."""
    import hashlib

    h = hashlib.new(algorithm)
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
