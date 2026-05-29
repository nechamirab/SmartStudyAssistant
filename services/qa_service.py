import re
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
        is_reliable: Whether the answer is grounded in the retrieved document.
    """
    query: str
    answer: str
    sources: List[str]
    is_reliable: bool = True


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
            is_reliable = self._is_answer_reliable(query, texts)

            if not is_reliable:
                missing_topic = self._extract_missing_topic(query)
                answer = (
                    f"I could not find a reliable answer about {missing_topic} "
                    "in the uploaded document."
                )
            else:
                answer = self._build_answer(query, texts)

            return QAResponse(
                query=query,
                answer=answer,
                sources=texts,
                is_reliable=is_reliable,
            )

        except Exception as e:
            raise QAError(f"QA failed: {e}") from e

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
        stopwords = {
            "what", "is", "a", "an", "the", "of", "in", "on",
            "for", "to", "about", "does", "do", "how", "why",
            "can", "should", "with", "as", "and", "or", "from",
            "that", "this", "are", "be", "it",
        }
        return [token for token in tokens if token not in stopwords]

    @staticmethod
    def _synonym_map() -> dict[str, List[str]]:
        return {
            "inner": ["inner join", "join"],
            "join": ["inner join", "join"],
            "sql": ["structured query language", "sql"],
            "rdbms": ["relational database management system", "relational database"],
            "select": ["select"],
            "table": ["table"],
        }

    @classmethod
    def _term_matches_text(cls, term: str, text: str) -> bool:
        if term in text:
            return True
        for synonym in cls._synonym_map().get(term, []):
            if synonym in text:
                return True
        return False

    @classmethod
    def _is_answer_reliable(cls, query: str, texts: List[str]) -> bool:
        if not texts:
            return False

        combined_text = "\n\n".join(texts).lower()
        keywords = cls._extract_keywords(query)
        if not keywords:
            return False

        # Retrieve phrase-specific requirements for high-risk topics like INNER JOIN
        if "inner join" in query.lower():
            if "inner join" not in combined_text and "join" not in combined_text:
                return False

        matching = 0
        for keyword in keywords:
            if cls._term_matches_text(keyword, combined_text):
                matching += 1

        coverage = matching / len(keywords)
        return coverage >= 0.4

    @staticmethod
    def _extract_missing_topic(query: str) -> str:
        normalized = query.strip().rstrip("? ").lower()
        if "inner join" in normalized:
            return "INNER JOIN"
        if "join" in normalized:
            return "JOIN"

        topic = re.sub(r"[^A-Za-z0-9 ]+", "", normalized).strip()
        if topic:
            return topic.upper()

        return "the requested topic"

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