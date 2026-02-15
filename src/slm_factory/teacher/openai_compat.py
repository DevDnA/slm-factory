"""OpenAI 호환 교사 백엔드(vLLM, LiteLLM, OpenAI 등)."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import TeacherConfig
from ..utils import get_logger
from .base import BaseTeacher

logger = get_logger("teacher.openai_compat")


class OpenAICompatTeacher(BaseTeacher):
    """OpenAI 호환 ``/v1/chat/completions`` API와 통신하는 교사입니다.

    vLLM, LiteLLM, OpenRouter 및 OpenAI 자체와 기본적으로 작동합니다.

    매개변수
    ----------
    config:
        ``backend="openai"``인 :class:`TeacherConfig`.
    """

    def __init__(self, config: TeacherConfig) -> None:
        self.model: str = config.model
        self.api_base: str = config.api_base.rstrip("/")
        self.api_key: str | None = config.api_key
        self.temperature: float = config.temperature
        self.timeout: int = config.timeout

        logger.info(
            "OpenAICompatTeacher initialised  model=%s  api_base=%s",
            self.model,
            self.api_base,
        )

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """API 키가 설정되었을 때 인증을 포함한 요청 헤더를 구성합니다."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------
    # BaseTeacher 인터페이스
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """*prompt*를 ``/v1/chat/completions``에 사용자 메시지로 전송합니다.

        키워드 인자는 ``model``과 ``temperature``를 오버라이드할 수 있습니다.
        """
        url = f"{self.api_base}/v1/chat/completions"

        model: str = kwargs.pop("model", self.model)
        temperature: float = kwargs.pop("temperature", self.temperature)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        logger.debug("POST %s  model=%s  temp=%.2f", url, model, temperature)

        try:
            resp = httpx.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.error("OpenAI-compat request timed out after %ds", self.timeout)
            raise RuntimeError(
                f"OpenAI-compat request timed out after {self.timeout}s "
                f"(model={model}, url={url})"
            ) from None
        except httpx.ConnectError:
            logger.error("Cannot connect to %s", self.api_base)
            raise RuntimeError(
                f"Cannot connect to OpenAI-compatible API at {self.api_base}. "
                "Is the server running?"
            ) from None
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenAI-compat HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise RuntimeError(
                f"OpenAI-compat HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        data: dict[str, Any] = resp.json()

        try:
            text: str = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected response structure: %s", data)
            raise RuntimeError(
                f"Unexpected response from {url}: {data!r:.300}"
            ) from exc

        if not text:
            logger.warning(
                "OpenAI-compat returned empty content for model=%s", model
            )

        return text

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """비동기 변형 — 동시 요청을 위해 ``httpx.AsyncClient``를 사용합니다."""
        url = f"{self.api_base}/v1/chat/completions"

        model: str = kwargs.pop("model", self.model)
        temperature: float = kwargs.pop("temperature", self.temperature)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(
                f"OpenAI-compat request timed out after {self.timeout}s "
                f"(model={model}, url={url})"
            ) from None
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to OpenAI-compatible API at {self.api_base}. "
                "Is the server running?"
            ) from None
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"OpenAI-compat HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        data: dict[str, Any] = resp.json()

        try:
            text: str = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response from {url}: {data!r:.300}"
            ) from exc

        if not text:
            logger.warning(
                "OpenAI-compat returned empty content for model=%s", model
            )

        return text

    def health_check(self) -> bool:
        """``/v1/models``를 핑하여 API에 도달할 수 있는지 확인합니다."""
        url = f"{self.api_base}/v1/models"
        try:
            resp = httpx.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            logger.debug("OpenAI-compat health-check OK")
            return True
        except (httpx.HTTPError, httpx.ConnectError):
            logger.warning("OpenAI-compat health-check FAILED at %s", url)
            return False
