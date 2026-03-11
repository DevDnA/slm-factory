"""ChromaDB 벡터 인덱싱 — corpus.parquet을 임베딩하여 ChromaDB에 적재합니다.

``AutoRAGExporter``가 생성한 ``corpus.parquet`` 파일을 읽어
sentence-transformers 모델로 임베딩한 뒤 ChromaDB에 upsert합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("rag.indexer")


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB 호환 메타데이터로 변환합니다.

    ChromaDB는 메타데이터 값으로 str, int, float, bool만 허용합니다.
    dict/list는 JSON 문자열로 변환하고, None 값은 제거합니다.
    """
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (dict, list)):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = str(value)
    return sanitized


class RAGIndexer:
    """corpus.parquet을 ChromaDB 벡터 DB에 임베딩하여 적재합니다."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.db_path = Path(config.paths.output) / config.rag.vector_db_path

    def index(self, corpus_path: Path) -> Path:
        """corpus.parquet을 읽어 ChromaDB에 임베딩 후 upsert합니다.

        매개변수
        ----------
        corpus_path:
            ``corpus.parquet`` 파일 경로.

        반환값
        -------
        Path
            ChromaDB 저장 경로.
        """
        import pandas as pd

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers가 설치되지 않았습니다. "
                "pip install sentence-transformers 로 설치하세요."
            )

        try:
            import chromadb
        except ImportError:
            raise RuntimeError(
                "chromadb가 설치되지 않았습니다. "
                "pip install chromadb 로 설치하세요."
            )

        # corpus.parquet 로드
        df = pd.read_parquet(corpus_path)
        logger.info("corpus.parquet 로드 완료 — %d개 청크", len(df))

        # 임베딩 모델 로드
        embedding_model = self.config.rag.embedding_model
        logger.info("임베딩 모델 로드: %s", embedding_model)
        model = SentenceTransformer(embedding_model)

        # ChromaDB 초기화
        self.db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.db_path))
        collection = client.get_or_create_collection(
            name=self.config.rag.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB 컬렉션 준비 완료: %s", self.config.rag.collection_name)

        # 데이터 추출
        doc_ids = df["doc_id"].tolist()
        contents = df["contents"].tolist()
        metadatas_raw = df["metadata"].tolist() if "metadata" in df.columns else [{}] * len(df)

        # 배치 단위 임베딩 및 upsert
        batch_size = 64
        total = len(contents)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_contents = contents[start:end]
            batch_ids = doc_ids[start:end]
            batch_metadatas = metadatas_raw[start:end]

            # 임베딩 생성
            embeddings = model.encode(batch_contents, show_progress_bar=False)
            embeddings_list = embeddings.tolist()

            # 메타데이터 정제
            sanitized_metadatas = []
            for meta in batch_metadatas:
                if isinstance(meta, dict):
                    sanitized_metadatas.append(_sanitize_metadata(meta))
                else:
                    sanitized_metadatas.append({})

            # ChromaDB upsert
            collection.upsert(
                ids=batch_ids,
                documents=batch_contents,
                embeddings=embeddings_list,
                metadatas=sanitized_metadatas,
            )
            logger.info(
                "임베딩 upsert 진행: %d/%d 청크 완료",
                end,
                total,
            )

        logger.info(
            "ChromaDB 인덱싱 완료 — %d개 청크, 저장 경로: %s",
            total,
            self.db_path,
        )
        return self.db_path
