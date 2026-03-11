"""AutoRAG 연동 데이터 내보내기 — 문서와 QA 쌍을 AutoRAG 평가용 parquet으로 변환합니다.

slm-factory의 ``parsed_documents.json``과 QA 쌍 파일(``qa_alpaca.json`` 등)을
AutoRAG가 요구하는 ``corpus.parquet`` + ``qa.parquet`` 형식으로 변환합니다.

AutoRAG 스키마:
- ``corpus.parquet``: doc_id(str), contents(str), metadata(dict)
- ``qa.parquet``: qid(str), query(str), retrieval_gt(list[list[str]]),
  generation_gt(list[str])
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("exporter.autorag_export")

# 결정적 UUID 생성을 위한 네임스페이스
_NAMESPACE = uuid.UUID("b7e23f8a-4c19-4d7b-9a31-0f8e6d2c5a47")


def _chunk_for_retrieval(content: str, chunk_size: int, overlap: int) -> list[str]:
    """검색용 문서 청킹을 수행합니다.

    문단 경계(``\\n\\n``)를 우선 존중하여 자연스러운 분할을 시도합니다.
    QA 생성용 청킹(``teacher.qa_generator.chunk_document``)과 동일한
    알고리즘을 사용하되, 검색에 최적화된 크기(기본 512자)로 분할합니다.
    """
    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = start + chunk_size

        if end >= len(content):
            chunks.append(content[start:])
            break

        boundary = content.rfind("\n\n", start + chunk_size // 2, end)
        if boundary > start:
            end = boundary

        chunks.append(content[start:end])
        new_start = end - overlap
        start = new_start if new_start > start else end

    return chunks


def _find_best_chunks(
    answer: str,
    chunk_texts: list[str],
    chunk_ids: list[str],
) -> list[str]:
    """답변 텍스트와 가장 많이 겹치는 청크 ID를 찾습니다.

    단어 수준 겹침과 부분 문자열 포함 여부를 함께 고려하여
    가장 관련성 높은 청크를 최대 3개까지 반환합니다.
    """
    if not chunk_texts:
        return []

    answer_words = set(answer.lower().split())
    scores: list[tuple[float, str]] = []

    for chunk_text, chunk_id in zip(chunk_texts, chunk_ids):
        chunk_words = set(chunk_text.lower().split())
        word_overlap = len(answer_words & chunk_words)
        # 답변 앞부분이 청크에 포함되면 보너스 부여
        containment = 1.0 if answer[:80].lower() in chunk_text.lower() else 0.0
        scores.append((word_overlap + containment * 20, chunk_id))

    scores.sort(reverse=True)

    top_score = scores[0][0]
    if top_score == 0:
        return [scores[0][1]]

    threshold = top_score * 0.5
    result = [cid for score, cid in scores if score >= threshold]
    return result[:3]


class AutoRAGExporter:
    """slm-factory 데이터를 AutoRAG 평가용 parquet 형식으로 변환합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.output_dir = Path(config.paths.output) / config.autorag_export.output_dir
        self.chunk_size = config.autorag_export.chunk_size
        self.overlap = config.autorag_export.overlap_chars

    def export(
        self,
        parsed_docs: list[dict[str, Any]],
        qa_pairs: list[dict[str, Any]],
    ) -> tuple[Path, Path]:
        """AutoRAG 호환 corpus.parquet + qa.parquet 파일을 생성합니다.

        매개변수
        ----------
        parsed_docs:
            ``ParsedDocument`` 딕셔너리 리스트
            (``parsed_documents.json``\\ 에서 로드).
        qa_pairs:
            QA 쌍 딕셔너리 리스트
            (``qa_alpaca.json`` 또는 ``qa_scored.json``\\ 에서 로드).

        반환값
        -------
        tuple[Path, Path]
            ``(corpus.parquet 경로, qa.parquet 경로)``
        """
        import pandas as pd  # datasets 의존성으로 이미 설치됨

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "AutoRAG 데이터 내보내기 시작 — 문서 %d건, QA %d건",
            len(parsed_docs),
            len(qa_pairs),
        )

        corpus_rows, doc_chunk_map = self._build_corpus(parsed_docs)
        logger.info("문서 청킹 완료 — %d개 청크 생성", len(corpus_rows))

        qa_rows = self._build_qa(qa_pairs, doc_chunk_map, corpus_rows)
        logger.info("QA 매핑 완료 — %d개 QA 항목 생성", len(qa_rows))

        corpus_path = self.output_dir / "corpus.parquet"
        qa_path = self.output_dir / "qa.parquet"

        pd.DataFrame(corpus_rows).to_parquet(corpus_path, index=False)
        pd.DataFrame(qa_rows).to_parquet(qa_path, index=False)

        logger.info(
            "AutoRAG 데이터 내보내기 완료 — %s, %s",
            corpus_path,
            qa_path,
        )
        return corpus_path, qa_path

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _build_corpus(
        self,
        parsed_docs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, str]]]]:
        """문서를 청킹하여 corpus 데이터와 문서→청크 매핑을 생성합니다.

        반환값
        -------
        tuple
            ``(corpus_rows, doc_chunk_map)``

            - ``corpus_rows``: AutoRAG corpus 행 리스트
            - ``doc_chunk_map``: ``{source_doc_id: [(chunk_doc_id, chunk_text), ...]}``
        """
        corpus_rows: list[dict[str, Any]] = []
        doc_chunk_map: dict[str, list[tuple[str, str]]] = {}

        for doc in parsed_docs:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            title = doc.get("title", "")
            metadata = doc.get("metadata", {})

            if not content or not content.strip():
                logger.debug("빈 문서 건너뜀: %s", doc_id)
                continue

            # 테이블 내용 병합
            tables = doc.get("tables", [])
            if tables:
                content = content + "\n\n" + "\n\n".join(tables)

            chunks = _chunk_for_retrieval(content, self.chunk_size, self.overlap)
            chunk_entries: list[tuple[str, str]] = []
            chunk_start = len(corpus_rows)

            offset = 0
            for i, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid5(_NAMESPACE, f"{doc_id}::{i}"))
                start_idx = offset
                end_idx = offset + len(chunk_text)
                # 다음 청크 시작점: 현재 청크 끝 - 중첩 영역
                offset = end_idx - self.overlap if i < len(chunks) - 1 else end_idx

                corpus_rows.append({
                    "doc_id": chunk_id,
                    "contents": chunk_text,
                    "path": metadata.get("path", title),
                    "start_end_idx": [start_idx, end_idx],
                    "metadata": {
                        "last_modified_datetime": datetime.now(),
                        "prev_id": None,
                        "next_id": None,
                        "source_doc_id": doc_id,
                        "source_title": title,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                })
                chunk_entries.append((chunk_id, chunk_text))

            # 동일 문서 내 청크 간 prev_id / next_id 설정
            for j in range(len(chunks)):
                idx = chunk_start + j
                if j > 0:
                    corpus_rows[idx]["metadata"]["prev_id"] = (
                        corpus_rows[idx - 1]["doc_id"]
                    )
                if j < len(chunks) - 1:
                    corpus_rows[idx]["metadata"]["next_id"] = (
                        corpus_rows[idx + 1]["doc_id"]
                    )

            doc_chunk_map[doc_id] = chunk_entries

        return corpus_rows, doc_chunk_map

    def _build_qa(
        self,
        qa_pairs: list[dict[str, Any]],
        doc_chunk_map: dict[str, list[tuple[str, str]]],
        corpus_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """QA 쌍을 AutoRAG 평가 데이터 형식으로 변환합니다.

        각 QA의 ``source_doc``을 기반으로 관련 코퍼스 청크를 찾아
        ``retrieval_gt`` (list[list[str]])를 생성합니다.
        """
        qa_rows: list[dict[str, Any]] = []
        all_chunk_ids = [row["doc_id"] for row in corpus_rows]

        for qa in qa_pairs:
            question = qa.get("question", qa.get("instruction", ""))
            answer = qa.get("answer", qa.get("output", ""))
            source_doc = qa.get("source_doc", "")

            if not question or not answer:
                continue

            if source_doc and source_doc in doc_chunk_map:
                entries = doc_chunk_map[source_doc]
                chunk_ids = [cid for cid, _ in entries]
                chunk_texts = [ct for _, ct in entries]
                relevant_ids = _find_best_chunks(answer, chunk_texts, chunk_ids)
            else:
                relevant_ids = [all_chunk_ids[0]] if all_chunk_ids else []

            if not relevant_ids:
                logger.warning(
                    "QA에 대한 관련 청크를 찾을 수 없음: %s",
                    question[:50],
                )
                continue

            qa_rows.append({
                "qid": str(uuid.uuid4()),
                "query": question,
                "retrieval_gt": [relevant_ids],
                "generation_gt": [answer],
            })

        return qa_rows
