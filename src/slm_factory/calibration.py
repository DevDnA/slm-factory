"""문서 적응형 자동 캘리브레이션 — 청킹 및 QA 생성 최적화."""

from __future__ import annotations

import math
import re

from .utils import get_logger

logger = get_logger("calibration")


def auto_chunk_size(content: str, max_context_chars: int = 12000) -> int:
    """문서 구조를 분석하여 최적 chunk_size를 계산합니다.

    알고리즘:
    - 목표 청크 수: sqrt(doc_len / 2000) — 문서 길이에 비례하되 점진적 증가
    - 단락 밀도 보정: 짧은 단락(구조적/조밀) → 작은 청크, 긴 단락(서술) → 큰 청크
    - 상한: max_context_chars - 2000 (프롬프트 오버헤드 확보)
    """
    doc_len = len(content)
    ceiling = max_context_chars - 2000

    if doc_len <= ceiling:
        return ceiling

    paras = [p for p in content.split("\n\n") if len(p.strip()) > 20]
    avg_para = (sum(len(p) for p in paras) / len(paras)) if paras else 350

    # 조밀(avg ~200) → 0.7x, 보통(avg ~350) → 1.0x, 서술(avg ~500+) → 1.3x
    density_scale = max(0.7, min(1.3, avg_para / 350))

    target_chunks = max(3, math.ceil(math.sqrt(doc_len / 2000)))
    raw = int((doc_len / target_chunks) * density_scale)

    result = max(2000, min(raw, ceiling))
    logger.info(
        "Auto chunk_size: doc=%d chars, avg_para=%d, density=%.2f, target_chunks=%d → %d",
        doc_len,
        int(avg_para),
        density_scale,
        target_chunks,
        result,
    )
    return result


def auto_questions_per_chunk(chunk: str) -> int:
    """청크 내용의 정보 밀도를 분석하여 최적 질문 수를 계산합니다.

    지표:
    - 기본: 한국어 ~1200자당 1개 질문
    - 숫자 보너스: 날짜, 금액, 비율 등 (최대 +3)
    - 목록 보너스: 항목화된 구조 (최대 +2)
    """
    chunk_len = len(chunk)

    base = chunk_len / 1200

    numbers = len(re.findall(r"\d{4}[년.]|\d+[%원건명개호조억만대]|\d+[.,]\d+", chunk))
    number_bonus = min(numbers * 0.3, 3.0)

    list_items = len(
        re.findall(
            r"(?m)"
            r"^\s*[-•·◦▪]\s"
            r"|^\s*\d+[.)]\s"
            r"|^\s*[가-힣][.)]\s"
            r"|^\s*[oO]\s"
            r"|^\s*\(\d+\)",
            chunk,
        )
    )
    list_bonus = min(list_items * 0.2, 2.0)

    raw = base + number_bonus + list_bonus
    result = max(3, min(round(raw), 15))
    logger.debug(
        "Auto questions: %d chars, base=%.1f, num_bonus=%.1f, list_bonus=%.1f → %d",
        chunk_len,
        base,
        number_bonus,
        list_bonus,
        result,
    )
    return result
