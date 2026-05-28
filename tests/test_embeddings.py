import unittest

from embeddings.providers import EmbeddingProviderError, EmbeddingProviderRegistry
from services.embedding_service import EmbeddingService


class EmbeddingTests(unittest.TestCase):
    def test_mock_provider_is_deterministic(self):
        provider = EmbeddingProviderRegistry.create("mock", cache_enabled=False)

        self.assertEqual(provider.embed_query("alpha"), provider.embed_query("alpha"))

    def test_minilm_alias_resolves_without_loading_model(self):
        provider = EmbeddingProviderRegistry.create(
            "minilm",
            cache_enabled=False,
        )

        self.assertEqual(provider.provider_name, "sentence-transformers")
        self.assertIn("all-MiniLM", provider.model_name)

    def test_service_records_dimension(self):
        class Chunk:
            chunk_id = "a"
            text = "alpha"

        service = EmbeddingService(provider="mock", model="", cache_enabled=False)
        service.embed_texts([Chunk()])

        self.assertGreater(service.embedding_dimension, 0)

    def test_service_falls_back_to_mock_when_optional_provider_fails(self):
        class Chunk:
            chunk_id = "a"
            text = "alpha"

        class BrokenProvider:
            provider_name = "sentence-transformers"
            model_name = "broken"

            def embed_texts(self, texts, batch_size=32):
                raise EmbeddingProviderError("missing optional dependency")

        service = EmbeddingService(provider="mock", model="", cache_enabled=False)
        service.provider = "sentence-transformers"
        service._provider = BrokenProvider()

        embeddings = service.embed_texts([Chunk()])

        self.assertEqual(service.provider, "mock")
        self.assertEqual(len(embeddings[0].vector), service.embedding_dimension)


if __name__ == "__main__":
    unittest.main()
