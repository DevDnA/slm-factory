"""FastAPI RAG 서버 — ChromaDB 검색 + Ollama 생성으로 RAG 응답을 제공합니다.

ChromaDB에 적재된 벡터 DB를 검색하고, Ollama SLM에 컨텍스트와 함께
질문을 전달하여 문서 기반 답변을 생성하는 REST API 서버입니다.
"""

import json
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SLMConfig

from ..utils import get_logger

logger = get_logger("rag.server")

_RAG_SYSTEM_PROMPT = (
    "다음 문서를 참고하여 질문에 답변하십시오. "
    "문서에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답변하십시오."
)


def create_app(config: "SLMConfig"):
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
    import uuid
    from pathlib import Path

    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.responses import JSONResponse
    except ImportError:
        raise RuntimeError(
            "fastapi가 설치되지 않았습니다. pip install fastapi 로 설치하세요."
        )

    try:
        import chromadb
    except ImportError:
        raise RuntimeError(
            "chromadb가 설치되지 않았습니다. pip install chromadb 로 설치하세요."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers가 설치되지 않았습니다. "
            "pip install sentence-transformers 로 설치하세요."
        )

    import httpx
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
        stream: bool = False

    class QueryResponse(BaseModel):
        """RAG 질의 응답."""

        answer: str
        sources: list[Source]
        query: str

    # ------------------------------------------------------------------
    # 설정값 추출
    # ------------------------------------------------------------------

    db_path = Path(config.paths.output) / config.rag.vector_db_path
    ollama_model = config.rag.ollama_model or config.export.ollama.model_name
    api_base = config.teacher.api_base

    # ------------------------------------------------------------------
    # 라이프사이클 관리
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        collection = chroma_client.get_collection(
            name=config.rag.collection_name,
        )
        logger.info(
            "ChromaDB 컬렉션 로드 완료: %s (%d개 문서)",
            config.rag.collection_name,
            collection.count(),
        )

        embedding_model = SentenceTransformer(config.rag.embedding_model)
        logger.info("임베딩 모델 로드 완료: %s", config.rag.embedding_model)

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.rag.request_timeout),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
        logger.info("Ollama 모델: %s, API: %s", ollama_model, api_base)

        _app.state.collection = collection
        _app.state.chroma_client = chroma_client
        _app.state.embedding_model = embedding_model
        _app.state.http_client = http_client

        yield

        await http_client.aclose()
        logger.info("RAG 서버 리소스 정리 완료")

    # ------------------------------------------------------------------
    # FastAPI 앱 생성
    # ------------------------------------------------------------------

    app = FastAPI(
        title="slm-factory RAG 서비스",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.rag.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_tracking(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s — %.3fs [%s]",
            request.method,
            request.url.path,
            elapsed,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("처리되지 않은 예외 [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    # ------------------------------------------------------------------
    # 엔드포인트
    # ------------------------------------------------------------------

    def _search_documents(body: QueryRequest):
        collection = app.state.collection
        embedding_model = app.state.embedding_model

        top_k = body.top_k or config.rag.top_k
        query_embedding = embedding_model.encode(body.query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        sources: list[Source] = []
        context_parts: list[str] = []

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, doc_id, distance in zip(documents, ids, distances):
            score = max(0.0, min(1.0, 1.0 - distance))
            sources.append(Source(content=doc, doc_id=doc_id, score=score))
            context_parts.append(doc)

        context = "\n\n---\n\n".join(context_parts)
        prompt = f"{_RAG_SYSTEM_PROMPT}\n\n{context}\n\n질문: {body.query}\n답변:"
        return sources, prompt

    @app.post("/v1/query")
    async def query_rag(body: QueryRequest):
        """문서 검색 후 Ollama SLM으로 답변을 생성합니다.

        ``stream: true`` 요청 시 SSE(Server-Sent Events)로 토큰을 실시간 전송합니다.
        """
        from fastapi.responses import StreamingResponse

        sources, prompt = _search_documents(body)
        http_client = app.state.http_client

        if body.stream:

            async def _generate_stream():
                async with http_client.stream(
                    "POST",
                    f"{api_base}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": True,
                        "think": False,
                        "options": {"num_predict": config.rag.max_tokens},
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                        if chunk.get("done"):
                            break

                final = {
                    "sources": [s.model_dump() for s in sources],
                    "query": body.query,
                    "done": True,
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

            logger.info(
                "RAG 스트리밍 질의 시작 — 검색 %d건, 모델: %s",
                len(sources),
                ollama_model,
            )
            return StreamingResponse(
                _generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        response = await http_client.post(
            f"{api_base}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"num_predict": config.rag.max_tokens},
            },
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
            query=body.query,
        )

    @app.get("/health/live")
    async def health_live() -> dict:
        """활성 상태 프로브 — 서버가 실행 중이면 항상 200."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready() -> dict:
        """준비 상태 프로브 — ChromaDB 및 Ollama 연결을 확인합니다."""
        collection = app.state.collection
        http_client = app.state.http_client

        status: dict = {
            "status": "ok",
            "chromadb": {"collection": config.rag.collection_name, "count": 0},
            "ollama": {"model": ollama_model, "status": "unknown"},
        }

        try:
            status["chromadb"]["count"] = collection.count()
        except Exception:
            status["status"] = "degraded"
            status["chromadb"]["status"] = "error"

        try:
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

    @app.get("/health")
    async def health_check() -> dict:
        """하위 호환성을 위한 /health/ready 별칭."""
        return await health_ready()

    return app


def run_server(config: "SLMConfig") -> None:
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
        "RAG 서버 시작: %s:%d (workers=%d, log_level=%s)",
        config.rag.server_host,
        config.rag.server_port,
        config.rag.workers,
        config.rag.log_level,
    )
    uvicorn.run(
        app,
        host=config.rag.server_host,
        port=config.rag.server_port,
        workers=config.rag.workers,
        log_level=config.rag.log_level,
    )
