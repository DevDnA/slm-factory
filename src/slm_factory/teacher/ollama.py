"""Ollama 교사 백엔드 — 로컬 Ollama REST API를 호출합니다."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ..config import TeacherConfig
from ..utils import get_logger
from .base import BaseTeacher

logger = get_logger("teacher.ollama")

# Retry 설정
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # 초

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

        resp: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = httpx.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 요청 타임아웃 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama 요청이 {_MAX_RETRIES}회 시도 후에도 타임아웃되었습니다 "
                    f"(model={model}, url={url}, timeout={self.timeout}s). "
                    f"서버 상태를 확인하거나 teacher.timeout 값을 늘려보세요."
                ) from e
            except httpx.ConnectError as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 연결 실패 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama({self.api_base})에 {_MAX_RETRIES}회 시도 후에도 연결할 수 없습니다. "
                    "서버가 실행 중인지 확인하세요: ollama serve"
                ) from e
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ) from exc

        assert resp is not None
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

        resp: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, timeout=self.timeout)
                    resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 요청 타임아웃 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama 요청이 {_MAX_RETRIES}회 시도 후에도 타임아웃되었습니다 "
                    f"(model={model}, url={url}, timeout={self.timeout}s). "
                    f"서버 상태를 확인하거나 teacher.timeout 값을 늘려보세요."
                ) from e
            except httpx.ConnectError as e:
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama 연결 실패 (시도 %d/%d), %.0f초 후 재시도...",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama({self.api_base})에 {_MAX_RETRIES}회 시도 후에도 연결할 수 없습니다. "
                    "서버가 실행 중인지 확인하세요: ollama serve"
                ) from e
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ) from exc

        assert resp is not None
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
