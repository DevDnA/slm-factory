"""여러 모듈에서 공유하는 핵심 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """모든 문서 파서의 구조화된 출력입니다."""

    doc_id: str
    """파일명 기반 고유 식별자입니다."""

    title: str
    """문서 제목 (메타데이터 또는 파일명에서 추출)입니다."""

    content: str
    """마크다운 형식의 전체 추출 텍스트입니다."""

    tables: list[str] = field(default_factory=list)
    """마크다운 문자열로 추출된 표입니다."""

    metadata: dict = field(default_factory=dict)
    """임의의 메타데이터: 날짜, 저자, 페이지 수 등입니다."""


@dataclass
class QAPair:
    """메타데이터를 포함한 단일 질문-답변 쌍."""
    question: str
    answer: str
    instruction: str = ""  # Alpaca 'instruction' 필드
    source_doc: str = ""   # 이 쌍이 나온 문서
    category: str = ""     # 질문 카테고리
    is_augmented: bool = False
