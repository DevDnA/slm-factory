"""RAG 서비스 — ChromaDB 벡터 인덱싱 및 FastAPI 서빙."""


def __getattr__(name: str):
    if name == "RAGIndexer":
        from .indexer import RAGIndexer

        return RAGIndexer
    if name == "create_app":
        from .server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RAGIndexer", "create_app"]
