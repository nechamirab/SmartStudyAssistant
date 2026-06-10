import unittest

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult, EmbeddingService
from services.retrieval_service import RetrievalService
from vectorstores.factory import VectorStoreFactory


class RetrievalTests(unittest.TestCase):
    def test_retrieval_returns_ranked_chunks(self):
        chunks = [
            DocumentChunk("a", 1, "alpha source"),
            DocumentChunk("b", 1, "beta source"),
        ]
        store = VectorStoreFactory.create("memory")
        store.add(
            chunks,
            [
                EmbeddingResult("a", [1.0, 0.0]),
                EmbeddingResult("b", [0.0, 1.0]),
            ],
        )
        service = EmbeddingService(provider="mock", model="", cache_enabled=False)
        service.embed_query = lambda query: [1.0, 0.0]

        response = RetrievalService(service, store, chunks).retrieve("alpha", top_k=1)

        self.assertEqual(response.results[0].chunk.chunk_id, "a")


if __name__ == "__main__":
    unittest.main()
