from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core.models import DocumentChunk, DocumentPage
from services.exam_service import ExamOptions, ExamService
from services.general_ai_service import GeneralAIService
from services.pdf_render_service import PdfRenderService
from services.pdf_section_service import PdfSectionError, PdfSectionService
from services.pdf_service import PdfExtractionError, PdfService
from services.progress_service import ProgressService
from services.study_service import StudyService


def make_pdf_bytes() -> bytes:
    try:
        import fitz
    except ImportError:  # pragma: no cover
        raise unittest.SkipTest("PyMuPDF is not installed")

    doc = fitz.open()
    for page_number in range(1, 4):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Algorithms Section {page_number}\nThis page explains sorting, searching, and complexity analysis.",
        )
    data = doc.tobytes()
    doc.close()
    return data


class TestMVPRecovery(unittest.TestCase):
    def test_pdf_extraction_fallback_does_not_crash(self):
        pages = [DocumentPage(page_number=1, text="Fallback text", source_id="notes.pdf")]
        with patch.object(PdfService, "_extract_with_pymupdf", side_effect=RuntimeError("fitz failed")):
            with patch.object(PdfService, "_extract_with_pypdf", return_value=pages):
                service = PdfService()
                with patch("pathlib.Path.exists", return_value=True):
                    result = service.extract_pages("notes.pdf")
        self.assertEqual(result[0].text, "Fallback text")

    def test_section_pdf_cropping_valid_and_invalid_ranges(self):
        pdf_bytes = make_pdf_bytes()
        section = PdfSectionService.extract_section_pdf(pdf_bytes, 1, 2)
        self.assertGreater(len(section), 100)
        with self.assertRaises(PdfSectionError):
            PdfSectionService.extract_section_pdf(pdf_bytes, 3, 2)

    def test_pdf_rendering_returns_images_or_empty_list(self):
        pdf_bytes = make_pdf_bytes()
        images = PdfRenderService.render_pages(pdf_bytes, 1, 1)
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].startswith(b"\x89PNG"))
        self.assertEqual(PdfRenderService.render_pages(pdf_bytes, 10, 11), [])

    def test_study_plan_titles_and_next_section(self):
        pages = [
            DocumentPage(1, "INTRODUCTION TO DATABASES\nDatabases organize persistent records and queries."),
            DocumentPage(2, "SQL joins combine related rows using keys and predicates."),
            DocumentPage(3, "Indexes improve retrieval performance for common searches."),
        ]
        sections = StudyService().generate_study_plan(pages, pages_per_section=1)
        self.assertEqual(len(sections), 3)
        self.assertNotIn("page_1_chunk", sections[0].title.lower())
        self.assertGreaterEqual(sections[0].start_page, 1)
        self.assertEqual(StudyService.next_section_index(0, len(sections)), 1)
        self.assertEqual(StudyService.next_section_index(2, len(sections)), 2)

    def test_study_plan_rejects_generic_content_and_adds_objectives(self):
        pages = [
            DocumentPage(
                1,
                "Content\nWhat I have outlined above is the content from the source page.\n"
                "Recursion trees expand recurrence relations into levels of repeated subproblems. "
                "The method helps estimate total work by summing costs across each level.",
            )
        ]

        section = StudyService().generate_study_plan(pages, pages_per_section=1)[0]

        self.assertEqual(section.title, "Section 1: Recursion, Trees")
        self.assertNotIn("outlined above", section.summary.lower())
        self.assertGreaterEqual(len(section.learning_objectives), 3)
        self.assertIn("Recursion", section.key_concepts)

    def test_timer_logic_has_no_autorefresh_dependency(self):
        progress = ProgressService.default_state()
        ProgressService.start_timer(progress, now=100.0)
        ProgressService.pause_timer(progress, now=130.0)
        self.assertEqual(progress.actual_study_seconds, 30)
        self.assertFalse(progress.timer_running)

    def test_general_ai_provider_selection(self):
        service = GeneralAIService()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai", "GROQ_API_KEY": "groq"}, clear=True):
            self.assertEqual(service.select_provider().name, "openai")
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq"}, clear=True):
            self.assertEqual(service.select_provider().name, "groq")
        with patch.dict(os.environ, {}, clear=True):
            result = service.ask([], "How should I study?")
        self.assertFalse(result["ok"])
        self.assertIn("OPENAI_API_KEY", result["answer"])

    def test_source_labels_do_not_expose_chunk_ids(self):
        chunk = DocumentChunk(chunk_id="page_5_chunk_1", page_number=5, text="Evidence", source_id="notes.pdf")
        label = chunk.citation_label()
        self.assertIn("page 5", label)
        self.assertNotIn("chunk", label.lower())
        self.assertNotIn("page_5_chunk_1", label)

    def test_final_exam_malformed_ai_response_does_not_crash(self):
        with patch.object(ExamService, "_call_ai", return_value="not json"):
            exam = ExamService().generate_final_exam("Study context", ExamOptions(question_count=3))
        self.assertTrue(exam["fallback_used"])
        self.assertEqual(len(exam["questions"]), 3)

    def test_progress_cache_corruption_does_not_crash(self):
        progress = ProgressService.load("{not valid json")
        self.assertEqual(progress.completed_sections, set())
        self.assertEqual(progress.actual_study_seconds, 0)


if __name__ == "__main__":
    unittest.main()
