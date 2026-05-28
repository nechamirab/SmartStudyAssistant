import unittest

from core.models import DocumentPage
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.evaluation_service import EvaluationService
from services.vector_store_service import VectorStoreService


class TestCoreServices(unittest.TestCase):
    def test_chunk_service_splits_text(self):
        pages = [DocumentPage(page_number=1, text="This is a test document. " * 50)]
        chunk_service = ChunkService(chunk_size=100, chunk_overlap=20)
        chunks = chunk_service.chunk_pages(pages)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))

    def test_embedding_service_mock_vector(self):
        embedding_service = EmbeddingService(provider="mock")
        vector = embedding_service.embed_query("test sentence")

        self.assertEqual(len(vector), 128)
        self.assertAlmostEqual(sum(x * x for x in vector) ** 0.5, 1.0, places=6)

    def test_vector_store_search_returns_ranked_results(self):
        embedding_service = EmbeddingService(provider="mock")
        pages = [DocumentPage(page_number=1, text="First document."), DocumentPage(page_number=2, text="Second document.")]
        chunk_service = ChunkService(chunk_size=50, chunk_overlap=0)
        chunks = chunk_service.chunk_pages(pages)
        embeddings = embedding_service.embed_texts(chunks)

        vector_store = VectorStoreService()
        vector_store.add(chunks, embeddings)
        query_vector = embedding_service.embed_query("First")
        results = vector_store.search(query_vector, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_accuracy_token_f1(self):
        result = EvaluationService.calculate_accuracy_token_f1(
            "This is the correct answer.",
            "This is the correct answer.",
        )
        self.assertEqual(result, 1.0)

        result = EvaluationService.calculate_accuracy_token_f1(
            "This is somewhat correct.",
            "This is the correct answer.",
        )
        self.assertGreater(result, 0.0)


if __name__ == "__main__":
    unittest.main()
