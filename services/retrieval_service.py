from dataclasses import dataclass
from typing import List

from services.embedding_service import EmbeddingService, EmbeddingError
from services.vector_store_service import SearchResult, VectorStoreService


class RetrievalError(Exception):
    """Raised when retrieval fails."""
    pass


@dataclass(frozen=True)
class RetrievalResponse:
    """
    Represents the retrieval output for a user query.
    Attributes:
        query: The original user query.
        results: The most relevant search results.
    """
    query: str
    results: List[SearchResult]


class RetrievalService:
    """
    Service responsible for semantic retrieval.
    It converts the user query into an embedding vector and
    retrieves the most similar chunks from the vector store.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
        """
        Retrieve the most relevant chunks for a user query.
        Args:
            query: The user's question.
            top_k: Number of top results to return.
        Returns:
            RetrievalResponse containing the original query and matching chunks.
        Raises:
            RetrievalError: If embedding generation or search fails.
        """
        query = (query or "").strip()
        if not query:
            raise RetrievalError("Query cannot be empty.")

        if top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")

        try:
            query_vector = self.embedding_service.embed_query(query)
            results = self.vector_store.search(query_vector, top_k=top_k)
            return RetrievalResponse(query=query, results=results)
        except EmbeddingError as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}") from e