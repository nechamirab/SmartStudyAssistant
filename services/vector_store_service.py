from __future__ import annotations

from vectorstores.base import SearchResult
from vectorstores.memory import InMemoryVectorStore


class VectorStoreService(InMemoryVectorStore):
    """
    Backward-compatible alias for the default in-memory vector store.

    New code should prefer `vectorstores.VectorStoreFactory`, but the existing
    services import this class directly, so we keep the old name stable.
    """


__all__ = ["SearchResult", "VectorStoreService"]
