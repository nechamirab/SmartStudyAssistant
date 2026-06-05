import unittest

from core.config import NOT_FOUND_ANSWER
from core.models import DocumentPage
from services.rag_service import PDFRAGService
from services.source_utils import format_source_label, pdf_page_anchor


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

    def test_clean_source_label_hides_chunk_id_by_default(self):
        label = format_source_label(
            {
                "section_title": "Training Models",
                "page_number": 5,
                "chunk_id": "Ilovepdf_Merged_Pdf_Page_5_Chunk_1",
            }
        )

        self.assertEqual(label, "Source: Page 5")
        self.assertNotIn("Chunk", label)
        self.assertIn("chunk:", format_source_label({"page_number": 5, "chunk_id": "abc"}, debug=True))

    def test_section_pdf_anchor_opens_at_start_page(self):
        self.assertEqual(pdf_page_anchor(5), "#page=5&zoom=page-width")
        self.assertEqual(pdf_page_anchor(0), "#page=1&zoom=page-width")

    def test_retrieval_inspection_returns_scores_and_clean_labels(self):
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
                    page_number=5,
                    text="Training models requires evidence, evaluation, and source-grounded answers.",
                    source_id="notes.pdf",
                )
            ],
            "notes.pdf",
        )

        inspection = service.inspect_retrieval(index, "How are training models evaluated?", top_k=1)

        self.assertEqual(inspection["query"], "How are training models evaluated?")
        self.assertEqual(len(inspection["results"]), 1)
        self.assertIn("score", inspection["results"][0])
        self.assertEqual(inspection["results"][0]["source_label"], "Source: Page 5")
        self.assertNotIn("chunk", inspection["results"][0]["source_label"].lower())

    def test_not_found_answer_uses_requested_grounding_fallback(self):
        self.assertEqual(NOT_FOUND_ANSWER, "I could not find this clearly in the uploaded material.")


if __name__ == "__main__":
    unittest.main()
