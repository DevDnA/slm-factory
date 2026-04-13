"""인메모리 대화 세션 관리자입니다.

TTL 기반 자동 만료와 최대 턴 수 제한으로 메모리를 관리합니다.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ...utils import get_logger

logger = get_logger("rag.agent.session")


@dataclass
class Message:
    """대화 한 턴의 메시지."""

    role: str  # "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)


class SessionManager:
    """인메모리 세션 저장소 — TTL 기반 자동 만료.

    Parameters
    ----------
    ttl:
        세션 유지 시간(초). 마지막 접근 시점 기준.
    max_turns:
        세션당 보존할 최대 턴 수. 초과 시 오래된 턴 제거.
    """

    def __init__(self, ttl: int = 3600, max_turns: int = 20, max_sessions: int = 1000) -> None:
        self._sessions: dict[str, list[Message]] = {}
        self._last_access: dict[str, float] = {}
        self._ttl = ttl
        self._max_turns = max_turns
        self._max_sessions = max_sessions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """새 세션을 생성하고 ID를 반환합니다."""
        # 세션 수 제한 — 메모리 고갈 방지
        if len(self._sessions) >= self._max_sessions:
            self.cleanup_expired()
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._last_access, key=self._last_access.get)
                del self._sessions[oldest]
                del self._last_access[oldest]
                logger.debug("세션 수 제한(%d) 도달 — 최고령 세션 퇴거", self._max_sessions)

        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = []
        self._last_access[session_id] = time.time()
        return session_id

    def get_or_create(self, session_id: str | None) -> tuple[str, list[Message]]:
        """세션을 조회하거나, 없으면 새로 생성합니다.

        Returns
        -------
        tuple[str, list[Message]]
            (session_id, messages)
        """
        if session_id and session_id in self._sessions:
            self._last_access[session_id] = time.time()
            return session_id, self._sessions[session_id]

        new_id = self.create_session()
        return new_id, self._sessions[new_id]

    def add_message(self, session_id: str, message: Message) -> None:
        """세션에 메시지를 추가합니다."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(message)
        self._last_access[session_id] = time.time()

        # 최대 턴 수 초과 시 오래된 메시지 제거
        msgs = self._sessions[session_id]
        if len(msgs) > self._max_turns * 2:  # user+assistant 쌍
            excess = len(msgs) - self._max_turns * 2
            self._sessions[session_id] = msgs[excess:]

    def format_history(self, session_id: str) -> str:
        """프롬프트에 삽입할 대화 내역 텍스트를 생성합니다."""
        msgs = self._sessions.get(session_id, [])
        if not msgs:
            return ""

        lines: list[str] = ["[이전 대화]"]
        for msg in msgs:
            if msg.role == "user":
                lines.append(f"사용자: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"어시스턴트: {msg.content}")
        lines.append("")
        return "\n".join(lines)

    def cleanup_expired(self) -> int:
        """만료된 세션을 정리합니다.

        Returns
        -------
        int
            제거된 세션 수.
        """
        now = time.time()
        expired = [
            sid
            for sid, last in self._last_access.items()
            if now - last > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            del self._last_access[sid]

        if expired:
            logger.debug("만료 세션 %d개 정리 완료", len(expired))
        return len(expired)

    @property
    def active_count(self) -> int:
        """현재 활성 세션 수."""
        return len(self._sessions)
