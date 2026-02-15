"""Ollama 교사 백엔드 — 로컬 Ollama REST API를 호출합니다."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import TeacherConfig
from ..utils import get_logger
from .base import BaseTeacher

logger = get_logger("teacher.ollama")

# 사고 과정 추론을 유출하는 토큰 — 제거하거나 억제합니다.
_STOP_TOKENS: list[str] = [
    "</think>",
    "<think>",
    "Reasoning:",
    "Let me think",
    "Step by step:",
]


class OllamaTeacher(BaseTeacher):
    """Ollama 인스턴스와 통신하는 교사 구현입니다.

    매개변수
    ----------
    config:
        ``backend="ollama"``인 :class:`TeacherConfig`.
    """

    def __init__(self, config: TeacherConfig) -> None:
        self.model: str = config.model
        self.api_base: str = config.api_base.rstrip("/")
        self.temperature: float = config.temperature
        self.timeout: int = config.timeout

        logger.info(
            "OllamaTeacher initialised  model=%s  api_base=%s",
            self.model,
            self.api_base,
        )

    # ------------------------------------------------------------------
    # BaseTeacher 인터페이스
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """*prompt*를 Ollama ``/api/generate`` 엔드포인트로 전송합니다.

        키워드 인자는 ``model``, ``temperature``를 오버라이드하고
        ``format``을 추가할 수 있습니다(예: ``"json"``).
        """
        url = f"{self.api_base}/api/generate"

        model: str = kwargs.pop("model", self.model)
        temperature: float = kwargs.pop("temperature", self.temperature)
        fmt: str | None = kwargs.pop("format", None)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "stop": _STOP_TOKENS,
            },
        }
        if fmt is not None:
            payload["format"] = fmt

        logger.debug("POST %s  model=%s  temp=%.2f", url, model, temperature)

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %ds", self.timeout)
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s "
                f"(model={model}, url={url})"
            ) from None
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.api_base)
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.api_base}. "
                "Is the server running?"
            ) from None
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama returned HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise RuntimeError(
                f"Ollama HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        data: dict[str, Any] = resp.json()
        text: str = data.get("response", "").strip()

        if not text:
            logger.warning("Ollama returned an empty response for model=%s", model)

        return text

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """비동기 변형 — 동시 요청을 위해 ``httpx.AsyncClient``를 사용합니다."""
        url = f"{self.api_base}/api/generate"

        model: str = kwargs.pop("model", self.model)
        temperature: float = kwargs.pop("temperature", self.temperature)
        fmt: str | None = kwargs.pop("format", None)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "stop": _STOP_TOKENS,
            },
        }
        if fmt is not None:
            payload["format"] = fmt

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s "
                f"(model={model}, url={url})"
            ) from None
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.api_base}. "
                "Is the server running?"
            ) from None
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        data: dict[str, Any] = resp.json()
        text: str = data.get("response", "").strip()

        if not text:
            logger.warning("Ollama returned an empty response for model=%s", model)

        return text

    def health_check(self) -> bool:
        """``/api/tags``를 핑하여 Ollama에 도달할 수 있는지 확인합니다."""
        url = f"{self.api_base}/api/tags"
        try:
            resp = httpx.get(url, timeout=10)
            resp.raise_for_status()
            logger.debug("Ollama health-check OK")
            return True
        except (httpx.HTTPError, httpx.ConnectError):
            logger.warning("Ollama health-check FAILED at %s", url)
            return False
