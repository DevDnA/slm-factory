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

import re

from ..utils import get_logger

logger = get_logger("rag.server")

# Ollama thinking 모델(Qwen3 등)의 <think> 태그를 제거합니다.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_thinking_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()

_RAG_SYSTEM_PROMPT = (
    "당신은 문서 기반 전문 어시스턴트입니다. "
    "아래 [참고 문서]를 근거로 질문에 답변하세요.\n\n"
    "규칙:\n"
    "1. 답변의 핵심 사실마다 어떤 문서에서 왔는지 명시하세요 (예: '문서 2에 따르면...').\n"
    "2. 여러 문서의 정보를 종합하되, 문서 간 내용이 상충하면 차이점을 명확히 설명하세요.\n"
    "3. 문서에 근거한 구체적 수치, 날짜, 명칭을 우선 사용하세요.\n"
    "4. 문서에 직접적 근거가 없으면 '해당 정보를 찾을 수 없습니다'라고 답변하세요.\n"
    "5. 표, 목록, 소제목 등 마크다운을 활용하여 구조적으로 정리하세요.\n"
    "6. 답변은 완결성 있게 작성하되 간결하게 핵심만 전달하세요."
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
        try:
            count = qdrant_client.count(collection_name=collection_name).count
        except (ValueError, Exception):
            qdrant_client.close()
            raise RuntimeError(
                f"Qdrant 컬렉션 '{collection_name}'을(를) 찾을 수 없습니다.\n"
                f"  인덱스가 손상되었거나 구축이 완료되지 않았을 수 있습니다.\n"
                f"  해결: rm -rf {db_path} 후 slf rag를 다시 실행하세요."
            )
        if count == 0:
            qdrant_client.close()
            raise RuntimeError(
                f"Qdrant 컬렉션 '{collection_name}'이 비어 있습니다.\n"
                f"  해결: rm -rf {db_path} 후 slf rag를 다시 실행하세요."
            )
        logger.info(
            "Qdrant 컬렉션 로드 완료: %s (%d개 문서)",
            collection_name,
            count,
        )

        embedding_model = SentenceTransformer(config.rag.embedding_model)
        logger.info(
            "임베딩 모델 로드 완료: %s (device=%s)",
            config.rag.embedding_model,
            embedding_model.device.type,
        )

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
        _app.state.bm25_index = None
        _app.state.bm25_docs = None
        _app.state.bm25_ids = None
        _app.state.bm25_metadatas = None

        if config.rag.reranker_enabled:
            try:
                from sentence_transformers import CrossEncoder

                _app.state.reranker = CrossEncoder(
                    config.rag.reranker_model,
                    max_length=512,
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

                # BM25Okapi 인덱스 구축 (kiwi 형태소 토크나이저)
                try:
                    from rank_bm25 import BM25Okapi
                except ImportError:
                    raise RuntimeError(
                        "rank_bm25가 설치되지 않았습니다. "
                        "uv sync --extra rag 로 설치하세요."
                    )
                tokenized_corpus = [
                    _korean_tokenize(doc) for doc in _app.state.bm25_docs
                ]
                _app.state.bm25_index = BM25Okapi(tokenized_corpus)
                logger.info(
                    "BM25Okapi 인덱스 구축 완료: %d개 문서 (kiwi 형태소 토크나이저)",
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

    _kiwi_instance = None

    def _korean_tokenize(text: str) -> list[str]:
        nonlocal _kiwi_instance
        if _kiwi_instance is None:
            try:
                from kiwipiepy import Kiwi

                _kiwi_instance = Kiwi()
            except ImportError:
                _kiwi_instance = False
        if _kiwi_instance is False:
            return text.lower().split()
        tokens = _kiwi_instance.tokenize(text)  # type: ignore[union-attr]
        return [t.form for t in tokens if len(t.form) > 1 or not t.tag.startswith("J")]

    def _bm25_search(query: str, top_k: int) -> list[tuple[str, str, float, dict]]:
        """BM25Okapi를 사용한 키워드 기반 문서 검색입니다.

        kiwi 형태소 토크나이저로 쿼리를 분석하고, 사전 구축된
        BM25Okapi 인덱스에서 TF-IDF 기반 검색을 수행합니다.
        """
        bm25_index = app.state.bm25_index
        if bm25_index is None or not app.state.bm25_docs:
            return []
        query_tokens = _korean_tokenize(query)
        if not query_tokens:
            return []
        scores = bm25_index.get_scores(query_tokens)
        # 점수 기준 상위 top_k개 추출 (점수 > 0인 것만)
        top_indices = scores.argsort()[::-1][:top_k]
        results: list[tuple[str, str, float, dict]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break
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
                rewrite_data = response.json()
                raw = rewrite_data.get("response", "")
                if not raw:
                    raw = rewrite_data.get("thinking", "")
                expanded = _clean_thinking_tags(raw).strip()
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

        if config.rag.hybrid_search and app.state.bm25_index is not None:
            # Reciprocal Rank Fusion (RRF): 벡터 검색과 BM25 결과를 통합합니다.
            # RRF 점수 = sum(1 / (k + rank_i)) — k=60이 표준값입니다.
            rrf_k = 60

            # 벡터 검색 결과에 RRF 점수 부여
            rrf_scores: dict[str, float] = {}
            rrf_data: dict[str, tuple[str, dict]] = {}  # doc_id → (doc, meta)
            for rank, (doc, doc_id, _dist, meta) in enumerate(
                zip(documents, ids, distances, metadatas)
            ):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    rrf_k + rank + 1
                )
                if doc_id not in rrf_data:
                    rrf_data[doc_id] = (doc, meta)

            # BM25 검색 결과에 RRF 점수 추가
            bm25_results = _bm25_search(body.query, top_k * 2)
            for rank, (doc, doc_id, _score, meta) in enumerate(bm25_results):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    rrf_k + rank + 1
                )
                if doc_id not in rrf_data:
                    rrf_data[doc_id] = (doc, meta)

            # RRF 점수 기준 정렬 후 상위 top_k개 선택
            sorted_ids = sorted(
                rrf_scores, key=lambda did: rrf_scores[did], reverse=True
            )[:top_k]
            documents = [rrf_data[did][0] for did in sorted_ids]
            ids = list(sorted_ids)
            distances = [1.0 - rrf_scores[did] for did in sorted_ids]
            metadatas = [rrf_data[did][1] for did in sorted_ids]

        # -- 유사도 필터링: min_score 미만 문서 제거 ---
        min_score = config.rag.min_score
        if min_score > 0:
            filtered = [
                (doc, did, dist, meta)
                for doc, did, dist, meta in zip(documents, ids, distances, metadatas)
                if max(0.0, min(1.0, 1.0 - dist)) >= min_score
            ]
            if filtered:
                documents, ids, distances, metadatas = (
                    [x[0] for x in filtered],
                    [x[1] for x in filtered],
                    [x[2] for x in filtered],
                    [x[3] for x in filtered],
                )

        # -- Lost-in-the-middle 재정렬: 관련도 높은 문서를 처음과 끝에 배치 ---
        if len(documents) >= 3:
            items = list(zip(documents, ids, distances, metadatas))
            front = [items[i] for i in range(0, len(items), 2)]
            back = [items[i] for i in range(1, len(items), 2)]
            reordered = front + list(reversed(back))
            documents = [x[0] for x in reordered]
            ids = [x[1] for x in reordered]
            distances = [x[2] for x in reordered]
            metadatas = [x[3] for x in reordered]

        sources: list[Source] = []
        context_parts: list[str] = []
        seen_parents: set[str] = set()

        doc_num = 0
        for doc, doc_id, distance, metadata in zip(
            documents, ids, distances, metadatas
        ):
            score = max(0.0, min(1.0, 1.0 - distance))
            parent = metadata.get("parent_content", doc) if metadata else doc
            sources.append(Source(content=doc, doc_id=doc_id, score=score))
            parent_key = parent[:100]
            if parent_key not in seen_parents:
                seen_parents.add(parent_key)
                doc_num += 1
                context_parts.append(f"[문서 {doc_num}]\n{parent}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"{_RAG_SYSTEM_PROMPT}\n\n"
            f"[참고 문서]\n{context}\n\n"
            f"질문: {body.query}\n답변:"
        )
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
                has_response_tokens = False
                thinking_buf: list[str] = []
                try:
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
                                has_response_tokens = True
                                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                            else:
                                thinking_token = chunk.get("thinking", "")
                                if thinking_token:
                                    thinking_buf.append(thinking_token)
                            if chunk.get("done"):
                                break
                except Exception as exc:
                    logger.error("Ollama 스트리밍 오류: %s", exc)
                    yield f"data: {json.dumps({'token': f'\\n\\n[오류] Ollama 응답 실패: {exc}'}, ensure_ascii=False)}\n\n"

                # Ollama 0.19.0 호환: response가 빈 경우 thinking 내용을 fallback으로 전송
                if not has_response_tokens and thinking_buf:
                    fallback = _clean_thinking_tags("".join(thinking_buf))
                    if fallback:
                        yield f"data: {json.dumps({'token': fallback}, ensure_ascii=False)}\n\n"

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

        import httpx

        try:
            response = await http_client.post(
                f"{api_base}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": -1,
                    "options": {"num_predict": config.rag.max_tokens},
                },
            )
            response.raise_for_status()
        except httpx.ConnectError:
            logger.error("Ollama 서버에 연결할 수 없습니다: %s", api_base)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ollama_connection_error",
                    "message": (
                        f"Ollama 서버({api_base})에 연결할 수 없습니다. "
                        "Ollama가 실행 중인지 확인하세요."
                    ),
                },
            )
        except httpx.TimeoutException:
            logger.error("Ollama 응답 시간 초과 (timeout: %ss)", config.rag.request_timeout)
            return JSONResponse(
                status_code=504,
                content={
                    "error": "ollama_timeout",
                    "message": (
                        "Ollama 응답 시간이 초과되었습니다. "
                        "project.yaml의 rag.request_timeout 값을 늘려보세요."
                    ),
                },
            )
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP 오류 %d: %s", exc.response.status_code, exc.response.text[:200])
            return JSONResponse(
                status_code=502,
                content={
                    "error": "ollama_http_error",
                    "message": (
                        f"Ollama 서버 오류 (HTTP {exc.response.status_code}). "
                        f"모델 '{ollama_model}'이 올바르게 로드되었는지 확인하세요."
                    ),
                },
            )
        data = response.json()
        answer = _clean_thinking_tags(data.get("response", ""))
        if not answer:
            answer = _clean_thinking_tags(data.get("thinking", ""))

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
            has_response_tokens = False
            thinking_buf: list[str] = []
            try:
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
                            has_response_tokens = True
                            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                        else:
                            thinking_token = chunk.get("thinking", "")
                            if thinking_token:
                                thinking_buf.append(thinking_token)
                        if chunk.get("done"):
                            break
            except Exception as exc:
                logger.error("Ollama 스트리밍 오류: %s", exc)
                yield f"data: {json.dumps({'type': 'token', 'content': f'\\n\\n[오류] Ollama 응답 실패: {exc}'}, ensure_ascii=False)}\n\n"

            # Ollama 0.19.0 호환: response가 빈 경우 thinking 내용을 fallback으로 전송
            if not has_response_tokens and thinking_buf:
                fallback = _clean_thinking_tags("".join(thinking_buf))
                if fallback:
                    yield f"data: {json.dumps({'type': 'token', 'content': fallback}, ensure_ascii=False)}\n\n"

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
