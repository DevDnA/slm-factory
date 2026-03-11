"""FastAPI RAG 서버 — ChromaDB 검색 + Ollama 생성으로 RAG 응답을 제공합니다.

ChromaDB에 적재된 벡터 DB를 검색하고, Ollama SLM에 컨텍스트와 함께
질문을 전달하여 문서 기반 답변을 생성하는 REST API 서버입니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("rag.server")

_RAG_SYSTEM_PROMPT = (
    "다음 문서를 참고하여 질문에 답변하십시오. "
    "문서에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답변하십시오."
)


def create_app(config: SLMConfig):
    """RAG API 서버를 위한 FastAPI 애플리케이션을 생성합니다.

    매개변수
    ----------
    config:
        slm-factory 프로젝트 설정. RAG, export, teacher 설정을 참조합니다.

    반환값
    -------
    FastAPI
        구성 완료된 FastAPI 애플리케이션 인스턴스.
    """
    from pathlib import Path

    try:
        from fastapi import FastAPI
    except ImportError:
        raise RuntimeError(
            "fastapi가 설치되지 않았습니다. "
            "pip install fastapi 로 설치하세요."
        )

    try:
        import chromadb
    except ImportError:
        raise RuntimeError(
            "chromadb가 설치되지 않았습니다. "
            "pip install chromadb 로 설치하세요."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers가 설치되지 않았습니다. "
            "pip install sentence-transformers 로 설치하세요."
        )

    from pydantic import BaseModel

    # ------------------------------------------------------------------
    # Pydantic 요청/응답 모델
    # ------------------------------------------------------------------

    class Source(BaseModel):
        """검색된 문서 소스 정보."""

        content: str
        doc_id: str
        score: float

    class QueryRequest(BaseModel):
        """RAG 질의 요청."""

        query: str
        top_k: int | None = None

    class QueryResponse(BaseModel):
        """RAG 질의 응답."""

        answer: str
        sources: list[Source]
        query: str

    # ------------------------------------------------------------------
    # 리소스 초기화
    # ------------------------------------------------------------------

    db_path = Path(config.paths.output) / config.rag.vector_db_path
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=config.rag.collection_name)
    logger.info(
        "ChromaDB 컬렉션 로드 완료: %s (%d개 문서)",
        config.rag.collection_name,
        collection.count(),
    )

    embedding_model = SentenceTransformer(config.rag.embedding_model)
    logger.info("임베딩 모델 로드 완료: %s", config.rag.embedding_model)

    # Ollama 모델 결정
    ollama_model = config.rag.ollama_model or config.export.ollama.model_name
    api_base = config.teacher.api_base
    logger.info("Ollama 모델: %s, API: %s", ollama_model, api_base)

    # ------------------------------------------------------------------
    # FastAPI 앱 생성
    # ------------------------------------------------------------------

    app = FastAPI(title="slm-factory RAG 서비스", version="0.1.0")

    @app.post("/v1/query", response_model=QueryResponse)
    async def query_rag(request: QueryRequest) -> QueryResponse:
        """문서 검색 후 Ollama SLM으로 답변을 생성합니다."""
        import httpx

        top_k = request.top_k or config.rag.top_k

        # 질의 임베딩
        query_embedding = embedding_model.encode(request.query).tolist()

        # ChromaDB 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # 소스 구성
        sources: list[Source] = []
        context_parts: list[str] = []

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, doc_id, distance in zip(documents, ids, distances):
            score = 1.0 - distance
            sources.append(Source(content=doc, doc_id=doc_id, score=score))
            context_parts.append(doc)

        # 컨텍스트 조합
        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"{_RAG_SYSTEM_PROMPT}\n\n{context}\n\n"
            f"질문: {request.query}\n답변:"
        )

        # Ollama 호출
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=config.teacher.timeout,
            )
            response.raise_for_status()
            answer = response.json().get("response", "")

        logger.info(
            "RAG 질의 처리 완료 — 검색 %d건, 모델: %s",
            len(sources),
            ollama_model,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
        )

    @app.get("/health")
    async def health_check() -> dict:
        """ChromaDB 및 Ollama 연결 상태를 확인합니다."""
        import httpx

        status: dict = {
            "status": "ok",
            "chromadb": {"collection": config.rag.collection_name, "count": 0},
            "ollama": {"model": ollama_model, "status": "unknown"},
        }

        # ChromaDB 상태
        try:
            status["chromadb"]["count"] = collection.count()
        except Exception:
            status["status"] = "degraded"
            status["chromadb"]["status"] = "error"

        # Ollama 상태
        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(
                    f"{api_base}/api/tags",
                    timeout=5.0,
                )
                resp.raise_for_status()
                status["ollama"]["status"] = "connected"
        except Exception:
            status["status"] = "degraded"
            status["ollama"]["status"] = "disconnected"

        return status

    return app


def run_server(config: SLMConfig) -> None:
    """RAG API 서버를 실행합니다.

    매개변수
    ----------
    config:
        slm-factory 프로젝트 설정. ``config.rag.server_host``와
        ``config.rag.server_port``로 서버 바인딩 주소를 결정합니다.
    """
    import uvicorn

    app = create_app(config)
    logger.info(
        "RAG 서버 시작: %s:%d",
        config.rag.server_host,
        config.rag.server_port,
    )
    uvicorn.run(app, host=config.rag.server_host, port=config.rag.server_port)
