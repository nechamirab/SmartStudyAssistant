import unittest

from core.config import NOT_FOUND_ANSWER
from core.models import DocumentPage
from services.rag_service import PDFRAGService


class RAGServiceTests(unittest.TestCase):
    def test_answer_uses_pdf_context_when_relevant(self):
        service = PDFRAGService(
            chunk_size=300,
            chunk_overlap=20,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        index = service.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text="Photosynthesis lets plants convert sunlight into chemical energy.",
                    source_id="biology.pdf",
                )
            ],
            "biology.pdf",
        )

        result = service.answer(index, "What does photosynthesis convert?")

        self.assertTrue(result.found)
        self.assertIn("Photosynthesis", result.answer)
        self.assertEqual(result.sources[0].pdf_name, "biology.pdf")

    def test_answer_returns_not_found_for_unrelated_question(self):
        service = PDFRAGService(
            chunk_size=300,
            chunk_overlap=20,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        index = service.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text="The document explains photosynthesis in plants.",
                    source_id="biology.pdf",
                )
            ],
            "biology.pdf",
        )

        result = service.answer(index, "Who invented the telephone?")

        self.assertFalse(result.found)
        self.assertEqual(result.answer, NOT_FOUND_ANSWER)
        self.assertEqual(result.sources, [])

    def test_multi_pdf_chunks_keep_distinct_source_ids(self):
        service = PDFRAGService(
            chunk_size=300,
            chunk_overlap=20,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        index = service.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text="Biology notes explain cells and membranes.",
                    source_id="biology.pdf",
                ),
                DocumentPage(
                    page_number=1,
                    text="History notes explain primary sources and timelines.",
                    source_id="history.pdf",
                ),
            ],
            "2 PDFs: biology.pdf, history.pdf",
        )

        self.assertEqual(index.chunk_count, 2)
        self.assertEqual({chunk.source_id for chunk in index.chunks}, {"biology.pdf", "history.pdf"})
        self.assertEqual(len({chunk.chunk_id for chunk in index.chunks}), 2)

    def test_index_summary_includes_extraction_counts(self):
        service = PDFRAGService(
            chunk_size=300,
            chunk_overlap=20,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        index = service.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text="Normal extracted text from a page.",
                    source_id="notes.pdf",
                    metadata={"extraction_method": "normal", "ocr_mode": "auto"},
                ),
                DocumentPage(
                    page_number=2,
                    text="OCR extracted text from a scanned page.",
                    source_id="notes.pdf",
                    metadata={"extraction_method": "ocr", "ocr_mode": "auto"},
                ),
            ],
            "notes.pdf",
        )

        extraction = index.to_summary()["extraction"]

        self.assertEqual(extraction["pages_processed"], 2)
        self.assertEqual(extraction["pages_using_normal_text"], 1)
        self.assertEqual(extraction["pages_using_ocr"], 1)
        self.assertGreater(extraction["total_characters_extracted"], 0)


if __name__ == "__main__":
    unittest.main()
