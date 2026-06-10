import unittest

from core.models import DocumentChunk
from services.qa_service import QAService
from services.retrieval_service import RetrievalResponse
from services.vector_store_service import SearchResult


class FakeRetrievalService:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
        return RetrievalResponse(query=query, results=self.results)


class TestQAService(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            DocumentChunk(
                chunk_id="1",
                page_number=1,
                text="SQL is a query language used to select and manage data.",
            ),
            DocumentChunk(
                chunk_id="2",
                page_number=1,
                text="Tables organize rows and columns in relational database systems.",
            ),
        ]
        self.results = [
            SearchResult(chunk=self.chunks[0], score=0.9),
            SearchResult(chunk=self.chunks[1], score=0.8),
        ]
        self.qa_service = QAService(FakeRetrievalService(self.results))

    def test_inner_join_returns_no_reliable_answer(self):
        response = self.qa_service.answer("What is inner join?")

        self.assertFalse(response.is_reliable)
        self.assertIn("could not find a reliable answer", response.answer.lower())
        self.assertEqual(response.sources, [chunk.text for chunk in self.chunks])

    def test_sql_question_remains_reliable(self):
        response = self.qa_service.answer("What is SQL?")

        self.assertTrue(response.is_reliable)
        self.assertIn("based on the document", response.answer.lower())
        self.assertEqual(response.sources, [chunk.text for chunk in self.chunks])


if __name__ == "__main__":
    unittest.main()
