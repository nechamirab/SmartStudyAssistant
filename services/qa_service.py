from dataclasses import dataclass
from typing import List

from services.retrieval_service import RetrievalService, RetrievalResponse


class QAError(Exception):
    """Raised when question answering fails."""
    pass


@dataclass(frozen=True)
class QAResponse:
    """
    Represents the answer generated for a user query.
    Attributes:
        query: The original user query.
        answer: The generated answer text.
        sources: The chunks used to build the answer.
    """
    query: str
    answer: str
    sources: List[str]


class QAService:
    """
    Simple QA service that builds an answer from retrieved chunks.
    """

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

    def answer(self, query: str) -> QAResponse:
        """
        Generate an answer based on retrieved chunks.
        Args:
            query: The user's question.
        Returns:
            QAResponse containing answer and sources.
        """
        if not query or not query.strip():
            raise QAError("Query cannot be empty.")

        try:
            retrieval_response = self.retrieval_service.retrieve(query, top_k=3)

            texts = [res.chunk.text for res in retrieval_response.results]

            answer = self._build_answer(query, texts)

            return QAResponse(
                query=query,
                answer=answer,
                sources=texts,
            )

        except Exception as e:
            raise QAError(f"QA failed: {e}") from e

    @staticmethod
    def _build_answer(query: str, texts: List[str]) -> str:
        """
        Build a simple answer from retrieved texts.

        NOTE:
        This is a placeholder implementation.
        In the future, this method will be replaced with an LLM-based solution
        (e.g., OpenAI) that generates a natural language answer using the retrieved context.
        """
        if not texts:
            return "No relevant information found."

        combined = "\n\n".join(texts[:3])

        return f"Based on the document:\n\n{combined[:500]}"