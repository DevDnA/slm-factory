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


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------

# 한국어 문서 섹션 헤더 패턴 (우선순위 순 — 상위가 더 큰 섹션)
_SECTION_PATTERNS = [
    # Level 1: 로마 숫자 대섹션 (Ⅰ. 제목, Ⅱ. 제목)
    re.compile(r"(?m)^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.．]\s*.+"),
    # Level 2: 아라비아 숫자 섹션 (1. 제목, 2. 제목) — "2026." 같은 연도 제외
    re.compile(r"(?m)^(?!(?:19|20)\d{2}[.．])\d{1,2}[.．]\s+\S.+"),
    # Level 3: 괄호 번호 (1), 2)) 또는 가., 나.
    re.compile(r"(?m)^(?:\(\d+\)|\d+\))\s+\S.+"),
    re.compile(r"(?m)^[가-힣][.．)]\s+\S.+"),
    # Level 4: 기호 (□, ○)
    re.compile(r"(?m)^[□■○●◦]\s+.+"),
]

_MIN_CHUNK_CHARS = 500  # 이보다 작은 청크는 다음과 병합
_MIN_USEFUL_CHARS = 200  # 이보다 작은 최종 청크는 제거 (목차/헤더만 있는 경우)


def _find_section_boundaries(content: str) -> list[int]:
    """문서에서 섹션 경계 위치를 찾습니다.

    가장 높은 레벨의 헤더를 먼저 사용합니다.
    해당 레벨에서 3개 이상의 섹션이 발견되면 그 레벨 사용.
    없으면 다음 레벨 시도.
    """
    for pattern in _SECTION_PATTERNS:
        matches = list(pattern.finditer(content))
        if len(matches) >= 3:
            return [m.start() for m in matches]
    return []


def _split_large_section(section: str, max_size: int) -> list[str]:
    """대형 섹션을 하위 헤더나 단락 경계에서 분할합니다."""
    # 하위 레벨 헤더로 분할 시도
    for pattern in _SECTION_PATTERNS:
        matches = list(pattern.finditer(section))
        if len(matches) >= 2:
            sub_boundaries = [m.start() for m in matches]
            sub_sections = []
            for i, start in enumerate(sub_boundaries):
                end = (
                    sub_boundaries[i + 1]
                    if i + 1 < len(sub_boundaries)
                    else len(section)
                )
                sub_sections.append(section[start:end].strip())

            # 병합하여 max_size 이내로
            result: list[str] = []
            buf = ""
            for ss in sub_sections:
                if not buf:
                    buf = ss
                elif len(buf) + len(ss) + 2 <= max_size:
                    buf = buf + "\n\n" + ss
                else:
                    result.append(buf)
                    buf = ss
            if buf:
                result.append(buf)

            # 여전히 max_size 초과하는 청크가 있으면 paragraph split
            final: list[str] = []
            for chunk in result:
                if len(chunk) <= max_size:
                    final.append(chunk)
                else:
                    from .teacher.qa_generator import chunk_document

                    final.extend(
                        chunk_document(chunk, max_size, min(500, max_size // 5))
                    )
            return final

    # 하위 헤더 없음 → paragraph splitting
    from .teacher.qa_generator import chunk_document

    return chunk_document(section, max_size, min(500, max_size // 5))


def section_aware_chunk(content: str, max_chunk_size: int) -> list[str]:
    """문서의 섹션 구조를 인식하여 논리적 단위로 청킹합니다.

    알고리즘:
    1. 섹션 헤더를 감지하여 문서를 논리 섹션으로 분할
    2. 소형 섹션(< _MIN_CHUNK_CHARS)을 다음 섹션과 병합
    3. 대형 섹션(> max_chunk_size)을 하위 헤더나 단락 경계에서 분할
    4. 저품질 청크(< _MIN_USEFUL_CHARS) 필터링
    5. 헤더가 감지되지 않으면 기존 paragraph-based chunking으로 폴백
    """
    # Step 1: 섹션 경계 찾기
    boundaries = _find_section_boundaries(content)

    if not boundaries:
        # 폴백: 기존 paragraph-based chunking
        from .teacher.qa_generator import chunk_document

        return chunk_document(content, max_chunk_size, min(500, max_chunk_size // 5))

    # Step 2: 경계로 섹션 분할
    sections: list[str] = []
    # 첫 경계 이전 텍스트 (서문/목차)
    if boundaries[0] > 0:
        preamble = content[: boundaries[0]].strip()
        if preamble:
            sections.append(preamble)

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
        section_text = content[start:end].strip()
        if section_text:
            sections.append(section_text)

    # Step 3: 소형 섹션 병합
    merged: list[str] = []
    buffer = ""
    for section in sections:
        if buffer:
            combined = buffer + "\n\n" + section
            if len(combined) <= max_chunk_size:
                buffer = combined
                continue
            else:
                merged.append(buffer)
                buffer = section if len(section) < _MIN_CHUNK_CHARS else ""
                if not buffer:
                    merged.append(section)
        elif len(section) < _MIN_CHUNK_CHARS:
            buffer = section
        else:
            merged.append(section)
    if buffer:
        if merged:
            # 마지막 버퍼를 이전 청크에 붙이기
            last = merged[-1]
            if len(last) + len(buffer) + 2 <= max_chunk_size:
                merged[-1] = last + "\n\n" + buffer
            else:
                merged.append(buffer)
        else:
            merged.append(buffer)

    # Step 4: 대형 섹션 분할
    final_chunks: list[str] = []
    for section in merged:
        if len(section) <= max_chunk_size:
            final_chunks.append(section)
        else:
            sub_chunks = _split_large_section(section, max_chunk_size)
            final_chunks.extend(sub_chunks)

    # Step 5: 저품질 필터링
    filtered = [c for c in final_chunks if len(c.strip()) >= _MIN_USEFUL_CHARS]

    if not filtered:
        # 모두 필터링됐으면 폴백
        from .teacher.qa_generator import chunk_document

        return chunk_document(content, max_chunk_size, min(500, max_chunk_size // 5))

    return filtered
