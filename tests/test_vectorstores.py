import tempfile
import unittest
import importlib.util
from pathlib import Path

from core.models import DocumentChunk
from services.embedding_service import EmbeddingResult
from vectorstores.factory import VectorStoreFactory


class InMemoryVectorStoreTests(unittest.TestCase):
    def _store_with_chunks(self):
        store = VectorStoreFactory.create("memory", collection_name="test")
        chunks = [
            DocumentChunk(
                chunk_id="a",
                page_number=1,
                text="alpha source",
                source_id="doc-a",
                metadata={"section": "intro"},
            ),
            DocumentChunk(
                chunk_id="b",
                page_number=2,
                text="beta source",
                source_id="doc-b",
                metadata={"section": "methods"},
            ),
        ]
        embeddings = [
            EmbeddingResult(chunk_id="a", vector=[1.0, 0.0]),
            EmbeddingResult(chunk_id="b", vector=[0.0, 1.0]),
        ]
        store.add(chunks, embeddings)
        return store

    def test_search_orders_by_cosine_similarity(self):
        store = self._store_with_chunks()

        results = store.search([0.9, 0.1], top_k=2)

        self.assertEqual([result.chunk.chunk_id for result in results], ["a", "b"])

    def test_search_supports_metadata_filter(self):
        store = self._store_with_chunks()

        results = store.search(
            [1.0, 0.0],
            top_k=2,
            metadata_filter={"section": "methods"},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "b")

    def test_add_replaces_existing_chunk_id(self):
        store = self._store_with_chunks()
        store.add(
            [
                DocumentChunk(
                    chunk_id="a",
                    page_number=3,
                    text="updated",
                    source_id="doc-a",
                )
            ],
            [EmbeddingResult(chunk_id="a", vector=[0.0, 1.0])],
        )

        self.assertEqual(len(store.chunks), 2)
        self.assertEqual(store.search([0.0, 1.0], top_k=1)[0].chunk.text, "updated")

    def test_delete_by_source_id(self):
        store = self._store_with_chunks()

        deleted = store.delete(source_id="doc-a")

        self.assertEqual(deleted, 1)
        self.assertEqual([chunk.chunk_id for chunk in store.chunks], ["b"])

    def test_save_and_load_round_trip(self):
        store = self._store_with_chunks()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vectors.json"
            store.save(path)

            loaded = VectorStoreFactory.create("memory", collection_name="test")
            loaded.load(path)

        self.assertEqual(loaded.stats().num_vectors, 2)
        self.assertEqual(loaded.search([1.0, 0.0], top_k=1)[0].chunk.chunk_id, "a")

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu is not installed")
    def test_faiss_search_orders_by_cosine_similarity(self):
        store = VectorStoreFactory.create("faiss", collection_name="test")
        chunks = [
            DocumentChunk(chunk_id="a", page_number=1, text="alpha", source_id="a.pdf"),
            DocumentChunk(chunk_id="b", page_number=1, text="beta", source_id="b.pdf"),
        ]
        store.add(
            chunks,
            [
                EmbeddingResult(chunk_id="a", vector=[1.0, 0.0]),
                EmbeddingResult(chunk_id="b", vector=[0.0, 1.0]),
            ],
        )

        results = store.search([0.9, 0.1], top_k=1)

        self.assertEqual(results[0].chunk.chunk_id, "a")


if __name__ == "__main__":
    unittest.main()
