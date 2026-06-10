from __future__ import annotations

from pathlib import Path

from vectorstores.base import BaseVectorStore, VectorStoreError
from vectorstores.memory import InMemoryVectorStore


class VectorStoreFactory:
    """Create vector store backends by name."""

    @staticmethod
    def create(
        backend: str = "memory",
        collection_name: str = "default",
        persist_path: str | Path | None = None,
    ) -> BaseVectorStore:
        normalized = (backend or "memory").strip().lower()
        if normalized in {"memory", "in-memory", "in_memory"}:
            return InMemoryVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
            )
        if normalized == "faiss":
            from vectorstores.faiss_store import FaissVectorStore

            return FaissVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
            )
        if normalized == "chroma":
            from vectorstores.chroma_store import ChromaVectorStore

            return ChromaVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
            )
        if normalized == "qdrant":
            from vectorstores.qdrant_store import QdrantVectorStore

            return QdrantVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
            )
        raise VectorStoreError(
            f"Unsupported vector store backend: {backend}. "
            "Supported backends: memory, faiss, chroma, qdrant."
        )
