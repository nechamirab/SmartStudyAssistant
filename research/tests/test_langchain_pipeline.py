from __future__ import annotations

import unittest
from unittest.mock import patch

from rag.langchain_pipeline import (
    LangChainDependencyError,
    LangChainRAGPipeline,
)


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        text = documents[0].page_content
        metadata = documents[0].metadata
        return [
            FakeDocument(text[:20], metadata),
            FakeDocument(text[20:], metadata),
        ]


class TestLangChainRAGPipeline(unittest.TestCase):
    def test_pipeline_can_initialize(self):
        pipeline = LangChainRAGPipeline()

        self.assertEqual(pipeline.embedding_model_name, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(pipeline.chunk_size, 500)
        self.assertEqual(pipeline.chunk_overlap, 50)
        self.assertEqual(pipeline.top_k, 3)

    def test_clean_text_removes_xml_and_noise(self):
        pipeline = LangChainRAGPipeline()
        cleaned = pipeline.clean_text(
            "  <tag>SQL</tag>   is useful  \n\n||| \nA\n\nINNER JOIN   combines rows. "
        )

        self.assertNotIn("<tag>", cleaned)
        self.assertNotIn("|||", cleaned)
        self.assertIn("SQL is useful", cleaned)
        self.assertIn("INNER JOIN combines rows.", cleaned)

    def test_split_documents_creates_chunks(self):
        pipeline = LangChainRAGPipeline(chunk_size=20, chunk_overlap=5)
        documents = [
            FakeDocument(
                "This is a long enough sentence to split into multiple sections for testing.",
                {"source": "sample.pdf", "page": 1},
            )
        ]

        with patch.object(LangChainRAGPipeline, "_get_document_class", return_value=FakeDocument):
            with patch.object(LangChainRAGPipeline, "_get_text_splitter_class", return_value=FakeSplitter):
                chunks = pipeline.split_documents(documents)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk.page_content for chunk in chunks))
        self.assertEqual(chunks[0].metadata["source"], "sample.pdf")

    def test_answer_question_returns_no_answer_when_topic_missing(self):
        pipeline = LangChainRAGPipeline()

        with patch.object(
            pipeline,
            "retrieve",
            return_value=[
                {"text": "SQL is a language for querying relational databases.", "source": "sql.pdf", "page": 2, "score": 0.1}
            ],
        ):
            response = pipeline.answer_question("What is INNER JOIN?")

        self.assertIn("could not find a reliable answer", response["answer"].lower())
        self.assertEqual(response["citations"], [])

    def test_answer_includes_citations_and_sources(self):
        pipeline = LangChainRAGPipeline()
        retrieved = [
            {
                "text": "SQL stands for Structured Query Language and is used to query relational databases.",
                "source": "database_notes.pdf",
                "page": 4,
                "score": 0.02,
            },
            {
                "text": "Structured Query Language is commonly abbreviated as SQL.",
                "source": "database_notes.pdf",
                "page": 4,
                "score": 0.03,
            },
        ]

        with patch.object(pipeline, "retrieve", return_value=retrieved):
            response = pipeline.answer_question("What is SQL?")

        self.assertIn("database_notes.pdf p.4", response["citations"])
        self.assertEqual(response["sources"], [{"source": "database_notes.pdf", "page": 4}])
        self.assertEqual(response["retrieved_chunks"], retrieved)

    def test_optional_dependency_error_is_friendly(self):
        pipeline = LangChainRAGPipeline()

        with patch.object(
            LangChainRAGPipeline,
            "_get_pdf_loader_class",
            side_effect=LangChainDependencyError("PyPDFLoader is unavailable."),
        ):
            with self.assertRaises(LangChainDependencyError):
                pipeline.load_pdf("missing.pdf")


if __name__ == "__main__":
    unittest.main()
