# rag/

RAG service pipeline. ChromaDB vector indexing from corpus.parquet + FastAPI REST API with Ollama SLM generation.

## STRUCTURE

```
rag/
├── __init__.py    # __getattr__ lazy loading: RAGIndexer, create_app
├── indexer.py     # RAGIndexer — corpus.parquet → sentence-transformers embedding → ChromaDB upsert
└── server.py      # create_app() + run_server() — FastAPI RAG API (query + health endpoints)
```

## HOW IT WORKS

1. `RAGIndexer.index(corpus_path)` reads `corpus.parquet` (from AutoRAGExporter), embeds via sentence-transformers, upserts to ChromaDB in 64-chunk batches.
2. `create_app(config)` builds a FastAPI app with `/v1/query` (POST) and `/health` (GET) endpoints.
3. Query flow: embed query → ChromaDB cosine search → top_k results → build context prompt → Ollama generate → return answer + sources.
4. CLI commands `rag-index` and `rag-serve` in `cli.py` invoke these directly.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Change embedding model | Via `RagConfig.embedding_model` | Default: `BAAI/bge-m3` |
| Change vector DB path | Via `RagConfig.vector_db_path` | Relative to `paths.output` |
| Change search results count | Via `RagConfig.top_k` | Default: 5 |
| Change RAG prompt | `server.py:_RAG_SYSTEM_PROMPT` | Korean system prompt for grounded answers |
| Change batch size for indexing | `indexer.py` line 102 | Hardcoded 64 — consider making configurable |
| Change server host/port | Via `RagConfig.server_host/server_port` | Default: 0.0.0.0:8000 |
| Change Ollama model for RAG | Via `RagConfig.ollama_model` | Falls back to `export.ollama.model_name` |

## CONVENTIONS

- `__getattr__` lazy loading in `__init__.py` — same pattern as root package.
- All three heavy deps (`chromadb`, `sentence-transformers`, `fastapi`) are optional — `RuntimeError` with install hint on missing import.
- `_sanitize_metadata()` in indexer.py converts non-primitive metadata values for ChromaDB compatibility (dict/list → JSON string, None removed).
- `TYPE_CHECKING` import for `SLMConfig` to avoid circular deps at runtime.
- Pydantic request/response models defined inside `create_app()` — not module-level (avoids import-time FastAPI dependency).
- `httpx.AsyncClient` used for Ollama calls within the FastAPI async endpoint.

## ANTI-PATTERNS

- Do NOT import `chromadb`, `sentence_transformers`, or `fastapi` at module level — they are optional deps.
- Do NOT use this module from Pipeline — it's standalone CLI-only (`rag-index`, `rag-serve` commands).
- Do NOT change `collection.upsert()` to `collection.add()` — upsert enables idempotent re-indexing.
