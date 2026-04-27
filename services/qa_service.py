from dataclasses import dataclass
from typing import List

from services.retrieval_service import RetrievalService
from services.llm_service import LLMService, LLMError

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

    def __init__(
            self,
            retrieval_service: RetrievalService,
            llm_service: LLMService | None = None,
    ):
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service or LLMService()

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

            prompt = self._build_prompt(query, texts)
            answer = self.llm_service.generate(prompt)

            return QAResponse(
                query=query,
                answer=answer,
                sources=texts,
            )
        except LLMError as e:
            raise QAError(f"LLM answer generation failed: {e}") from e
        except Exception as e:
            raise QAError(f"QA failed: {e}") from e

    @staticmethod
    def _build_prompt(query: str, texts: List[str]) -> str:
        """
        Build a grounded QA prompt from retrieved texts.
        """
        if not texts:
            return (
                "You are a study assistant.\n"
                "Answer only based on the provided context.\n"
                "No relevant context was found.\n"
                "Say that the document does not provide enough information.\n\n"
                f"Question:\n{query}\n\n"
                "Answer:"
            )

        context = "\n\n---\n\n".join(texts[:3])

        return (
            "You are a study assistant answering questions about a PDF document.\n"
            "Use only the context below.\n"
            "Do not use outside knowledge.\n\n"
            "If the context gives a direct answer, answer clearly.\n"
            "If the context does not give a direct definition, explain what can be inferred "
            "from the context and clearly state that the document does not provide an explicit definition.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

