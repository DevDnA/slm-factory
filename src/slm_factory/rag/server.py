"""FastAPI RAG 서버 — Qdrant 검색 + Ollama 생성으로 RAG 응답을 제공합니다.

Qdrant에 적재된 벡터 DB를 검색하고, Ollama SLM에 컨텍스트와 함께
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
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError:
        raise RuntimeError(
            "fastapi가 설치되지 않았습니다. uv sync --extra rag 로 설치하세요."
        )

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise RuntimeError(
            "qdrant-client가 설치되지 않았습니다. uv sync --extra rag 로 설치하세요."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers가 설치되지 않았습니다. "
            "uv sync --extra rag 로 설치하세요."
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
        qdrant_client = QdrantClient(path=str(db_path))
        collection_name = config.rag.collection_name
        count = qdrant_client.count(collection_name=collection_name).count
        logger.info(
            "Qdrant 컬렉션 로드 완료: %s (%d개 문서)",
            collection_name,
            count,
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

        _app.state.qdrant_client = qdrant_client
        _app.state.collection_name = collection_name
        _app.state.embedding_model = embedding_model
        _app.state.http_client = http_client
        _app.state.reranker = None
        _app.state.bm25_docs = None
        _app.state.bm25_ids = None
        _app.state.bm25_metadatas = None

        if config.rag.reranker_enabled:
            try:
                from sentence_transformers import CrossEncoder

                _app.state.reranker = CrossEncoder(
                    config.rag.reranker_model, max_length=512
                )
                logger.info("Reranker 모델 로드 완료: %s", config.rag.reranker_model)
            except Exception:
                logger.warning("Reranker 로드 실패 — 비활성 상태로 계속합니다")

        if config.rag.hybrid_search:
            try:
                all_points = []
                offset = None
                while True:
                    points, next_offset = qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False,
                        offset=offset,
                    )
                    all_points.extend(points)
                    if next_offset is None:
                        break
                    offset = next_offset
                _app.state.bm25_docs = [
                    p.payload.get("document", "") for p in all_points
                ]
                _app.state.bm25_ids = [
                    p.payload.get("doc_id", str(p.id)) for p in all_points
                ]
                _app.state.bm25_metadatas = [
                    {
                        k: v
                        for k, v in p.payload.items()
                        if k not in ("document", "doc_id")
                    }
                    for p in all_points
                ]
                logger.info(
                    "BM25 인덱스 구축 완료: %d개 문서",
                    len(_app.state.bm25_docs),
                )
            except Exception:
                logger.warning("BM25 인덱스 구축 실패 — 비활성 상태로 계속합니다")

        port = config.rag.server_port
        logger.info(
            "\n\n"
            "  ╔══════════════════════════════════════════╗\n"
            "  ║  RAG 채팅 서비스 준비 완료!              ║\n"
            "  ║  http://localhost:%d/chat              ║\n"
            "  ╚══════════════════════════════════════════╝\n",
            port,
        )

        yield

        await http_client.aclose()
        qdrant_client.close()
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

    def _bm25_search(query: str, top_k: int) -> list[tuple[str, str, float, dict]]:
        """키워드 매칭 기반 검색 (BM25 대용)."""
        if not app.state.bm25_docs:
            return []
        query_terms = set(query.lower().split())
        if not query_terms:
            return []
        scores: list[tuple[float, int]] = []
        for i, doc in enumerate(app.state.bm25_docs):
            doc_lower = doc.lower()
            score = sum(1 for t in query_terms if t in doc_lower) / len(query_terms)
            scores.append((score, i))
        scores.sort(reverse=True)
        results: list[tuple[str, str, float, dict]] = []
        for score, idx in scores[:top_k]:
            if score > 0:
                meta = (
                    app.state.bm25_metadatas[idx]
                    if app.state.bm25_metadatas and idx < len(app.state.bm25_metadatas)
                    else {}
                )
                results.append(
                    (
                        app.state.bm25_docs[idx],
                        app.state.bm25_ids[idx],
                        score,
                        meta,
                    )
                )
        return results

    async def _rewrite_query(query: str) -> str:
        """짧은 질의를 검색에 적합한 형태로 확장합니다."""
        if len(query) > 30:
            return query
        http_client = app.state.http_client
        prompt = (
            f"다음 질의를 문서 검색에 적합하도록 확장하세요. "
            f"동의어, 관련 용어, 구체적 표현을 추가하세요. "
            f"확장된 질의만 반환하세요.\n\n"
            f"원본: {query}\n확장:"
        )
        try:
            response = await http_client.post(
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 100},
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                expanded = response.json().get("response", "").strip()
                if expanded:
                    return f"{query} {expanded}"
        except Exception:
            logger.debug("Query rewriting 실패 — 원본 질의를 사용합니다")
        return query

    def _search_documents(body: QueryRequest):
        qdrant_client = app.state.qdrant_client
        collection_name = app.state.collection_name
        embedding_model = app.state.embedding_model

        top_k = body.top_k or config.rag.top_k

        reranker = app.state.reranker
        use_reranker = reranker is not None
        initial_k = top_k * 3 if use_reranker else top_k

        query_embedding = embedding_model.encode(
            body.query, prompt_name="query"
        ).tolist()
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=initial_k,
            with_payload=True,
        )

        documents = [p.payload.get("document", "") for p in results.points]
        ids = [p.payload.get("doc_id", str(p.id)) for p in results.points]
        distances = [1.0 - p.score for p in results.points]
        metadatas = [
            {k: v for k, v in p.payload.items() if k not in ("document", "doc_id")}
            for p in results.points
        ]

        if use_reranker and documents:
            pairs = [(body.query, doc) for doc in documents]
            rerank_scores = reranker.predict(pairs)
            ranked = sorted(
                zip(rerank_scores, documents, ids, distances, metadatas),
                reverse=True,
            )
            ranked = ranked[:top_k]
            if ranked:
                _, documents, ids, distances, metadatas = zip(*ranked)
                documents = list(documents)
                ids = list(ids)
                distances = list(distances)
                metadatas = list(metadatas)
            else:
                documents, ids, distances, metadatas = [], [], [], []
        else:
            documents = documents[:top_k]
            ids = ids[:top_k]
            distances = distances[:top_k]
            metadatas = metadatas[:top_k]

        if config.rag.hybrid_search and app.state.bm25_docs:
            bm25_results = _bm25_search(body.query, top_k)
            seen_ids = set(ids)
            for doc, doc_id, score, meta in bm25_results:
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    documents.append(doc)
                    ids.append(doc_id)
                    distances.append(1.0 - score)
                    metadatas.append(meta)

        sources: list[Source] = []
        context_parts: list[str] = []
        seen_parents: set[str] = set()
        max_context_chars = 3000

        for doc, doc_id, distance, metadata in zip(
            documents, ids, distances, metadatas
        ):
            score = max(0.0, min(1.0, 1.0 - distance))
            parent = metadata.get("parent_content", doc) if metadata else doc
            sources.append(Source(content=doc, doc_id=doc_id, score=score))
            parent_key = parent[:100]
            if parent_key not in seen_parents:
                seen_parents.add(parent_key)
                context_parts.append(parent)

        context = "\n\n---\n\n".join(context_parts)
        if len(context) > max_context_chars:
            context = context[:max_context_chars]
        prompt = f"{_RAG_SYSTEM_PROMPT}\n\n{context}\n\n질문: {body.query}\n답변:"
        return sources, prompt

    @app.post("/v1/query")
    async def query_rag(body: QueryRequest):
        """문서 검색 후 Ollama SLM으로 답변을 생성합니다.

        ``stream: true`` 요청 시 SSE(Server-Sent Events)로 토큰을 실시간 전송합니다.
        """
        from fastapi.responses import StreamingResponse

        if config.rag.query_rewriting:
            expanded = await _rewrite_query(body.query)
            body_for_search = QueryRequest(
                query=expanded, top_k=body.top_k, stream=body.stream
            )
        else:
            body_for_search = body

        sources, prompt = _search_documents(body_for_search)
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
                        "keep_alive": -1,
                        "options": {
                            "num_predict": config.rag.max_tokens,
                            "num_ctx": 8192,
                        },
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
                "keep_alive": -1,
                "options": {"num_predict": config.rag.max_tokens, "num_ctx": 8192},
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
        """준비 상태 프로브 — Qdrant 및 Ollama 연결을 확인합니다."""
        qdrant_client = app.state.qdrant_client
        collection_name = app.state.collection_name
        http_client = app.state.http_client

        status: dict = {
            "status": "ok",
            "qdrant": {"collection": config.rag.collection_name, "count": 0},
            "ollama": {"model": ollama_model, "status": "unknown"},
        }

        try:
            count_result = qdrant_client.count(collection_name=collection_name)
            status["qdrant"]["count"] = count_result.count
        except Exception:
            status["status"] = "degraded"
            status["qdrant"]["status"] = "error"

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

    # ------------------------------------------------------------------
    # 웹 채팅 UI & SSE 스트리밍
    # ------------------------------------------------------------------

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page():
        html_path = Path(__file__).parent / "static" / "chat.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.post("/v1/stream")
    async def query_stream(body: QueryRequest):
        if config.rag.query_rewriting:
            expanded = await _rewrite_query(body.query)
            body_for_search = QueryRequest(
                query=expanded, top_k=body.top_k, stream=body.stream
            )
        else:
            body_for_search = body

        sources, prompt = _search_documents(body_for_search)
        http_client = app.state.http_client

        async def _generate():
            async with http_client.stream(
                "POST",
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
                    "keep_alive": -1,
                    "options": {"num_predict": config.rag.max_tokens, "num_ctx": 8192},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                    if chunk.get("done"):
                        break

            yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        logger.info(
            "RAG SSE 스트리밍 시작 — 검색 %d건, 모델: %s",
            len(sources),
            ollama_model,
        )
        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
