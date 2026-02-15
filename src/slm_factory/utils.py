"""Rich를 사용한 구조화된 로깅 및 비동기 유틸리티."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable

    import httpx
    from rich.progress import Progress

T = TypeVar("T")

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


async def run_bounded(
    semaphore: asyncio.Semaphore,
    coro: Awaitable[T],
    progress: Progress,
    task_id: int,
) -> T:
    """세마포어 제한 하에 코루틴을 실행하고 진행률을 갱신합니다."""
    async with semaphore:
        result = await coro
        progress.advance(task_id)
        return result


async def ollama_generate(
    client: httpx.AsyncClient,
    api_base: str,
    model_name: str,
    question: str,
    timeout: float,
) -> str:
    """Ollama /api/generate 엔드포인트로 답변을 생성합니다."""
    resp = await client.post(
        f"{api_base}/api/generate",
        json={"model": model_name, "prompt": question, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")
