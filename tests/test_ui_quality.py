import unittest
from io import BytesIO
import inspect

from core.models import DocumentPage, StudySection
from services.pdf_render_service import render_pdf_pages_to_images
from services.pdf_section_service import create_section_pdf
from services.rag_service import PDFRAGService
from services.source_utils import format_source_label, sanitize_visible_text
from services.study_service import StudyService
from ui.streamlit_app import (
    calculate_elapsed_seconds,
    badge_class,
    next_section_index,
    render_pdf_viewer,
    render_navigation,
    render_study_plan_card,
    render_study_mode,
    set_active_page,
    section_progress_status,
    section_pdf_cache_key,
    study_mode_tabs,
    inject_custom_css,
)


class StudyModeQualityTests(unittest.TestCase):
    def _dirty_index(self):
        rag = PDFRAGService(
            chunk_size=320,
            chunk_overlap=20,
            min_score=0.0,
            embedding_provider="mock",
            vector_store_backend="memory",
        )
        return rag.build_index(
            [
                DocumentPage(
                    page_number=1,
                    text=(
                        "[Source: Page 1]\n"
                        "Introduction to Algorithms\n"
                        "Algorithms solve computational problems through precise steps. "
                        "Training models requires objective functions, gradients, and evaluation. "
                        "Gradient descent updates parameters to reduce prediction error."
                    ),
                    source_id="algorithms.pdf",
                )
            ],
            "algorithms.pdf",
        )

    def test_section_title_is_not_raw_source_label(self):
        section = StudyService().create_study_plan(self._dirty_index(), target_section_count=1)[0]

        self.assertNotIn("[Source:", section.title)
        self.assertNotIn("Page 1", section.title)
        self.assertNotEqual(section.title, "[Source: Page 1]")

    def test_exam_focus_has_clean_study_content(self):
        index = self._dirty_index()
        service = StudyService()
        section = service.create_study_plan(index, target_section_count=1)[0]
        focus = service.exam_focus(section, index)

        self.assertGreaterEqual(len(focus["important_points"]), 3)
        self.assertEqual(len(focus["possible_exam_questions"]), 3)
        self.assertTrue(focus["common_mistakes"])
        self.assertTrue(focus["key_terms"])
        visible = " ".join(
            focus["important_points"]
            + focus["possible_exam_questions"]
            + focus["common_mistakes"]
            + focus["key_terms"]
        )
        self.assertNotIn("[Source:", visible)
        self.assertNotIn("chunk_id", visible.lower())
        self.assertNotIn("algorithms.pdf", visible)

    def test_citations_are_clean(self):
        dirty = {"section_title": "[Source: Page 1]", "page_number": 1, "chunk_id": "abc123"}
        section_ref = {"section_title": "Section 1", "page_number": 2, "chunk_id": "abc123"}

        self.assertEqual(format_source_label(dirty), "Source: Page 1")
        self.assertEqual(format_source_label(section_ref), "Source: Page 2")
        self.assertNotIn("chunk", format_source_label(section_ref).lower())

    def test_pdf_viewer_uses_images_not_html_embeds(self):
        source = inspect.getsource(render_pdf_viewer)

        self.assertIn("render_pdf_pages_to_images", source)
        self.assertIn("st.image", source)
        self.assertNotIn("base64", source)
        self.assertNotIn("<iframe", source)
        self.assertNotIn("<object", source)
        self.assertNotIn("components.html", source)

    def test_study_mode_timer_does_not_autorefresh(self):
        source = inspect.getsource(render_study_mode)

        self.assertNotIn("st_autorefresh", source)
        self.assertNotIn("interval=1000", source)

    def test_quizzes_tab_is_hidden_from_main_navigation(self):
        self.assertEqual(
            study_mode_tabs(),
            ["Upload", "Study Plan", "Study Mode", "Ask AI", "Final Exam", "Dashboard"],
        )
        self.assertNotIn("Quizzes", study_mode_tabs())
        self.assertNotIn("RAG Check", study_mode_tabs())

    def test_main_study_features_exist_in_navigation(self):
        labels = study_mode_tabs()

        for label in ["Upload", "Study Plan", "Study Mode", "Ask AI", "Final Exam", "Dashboard"]:
            self.assertIn(label, labels)

    def test_css_injection_helper_is_importable(self):
        self.assertTrue(callable(inject_custom_css))

    def test_css_does_not_use_red_as_primary_style(self):
        source = inspect.getsource(inject_custom_css)

        self.assertIn("--study-primary", source)
        self.assertIn("--study-accent", source)
        self.assertNotIn("#dc2626", source.lower())
        self.assertNotIn("#ef4444", source.lower())

    def test_explanation_level_control_is_not_shown(self):
        source = inspect.getsource(render_study_mode)

        self.assertIn("Generate Explanation", source)
        self.assertNotIn("Explanation level", source)
        self.assertNotIn("Beginner", source)
        self.assertNotIn("University Student", source)
        self.assertNotIn("Exam Preparation", source)

    def test_study_plan_card_renders_required_data(self):
        source = inspect.getsource(render_study_plan_card)

        self.assertIn("Section {position + 1}", source)
        self.assertIn("page_range_label", source)
        self.assertIn("estimated_minutes", source)
        self.assertIn("Start session", source)
        self.assertIn("Study Mode", source)

    def test_navigation_is_state_backed_not_tabs(self):
        source = inspect.getsource(render_navigation)

        self.assertIn("active_page", source)
        self.assertNotIn("st.tabs", source)
        self.assertTrue(callable(set_active_page))

    def test_section_progress_status_and_badges_are_safe(self):
        section = self._section("section-1", 1, 3)

        self.assertEqual(section_progress_status(section, {}, None), "Not started")
        self.assertEqual(section_progress_status(section, {}, "section-1"), "In progress")
        self.assertEqual(section_progress_status(section, {"completed_sections": ["section-1"]}, None), "Completed")
        self.assertIn("badge-hard", badge_class("Hard"))

    def test_next_section_changes_index_and_page_range(self):
        sections = [
            self._section("section-1", 1, 8),
            self._section("section-2", 9, 14),
        ]

        next_index = next_section_index(0, len(sections))

        self.assertEqual(next_index, 1)
        self.assertEqual(sections[next_index].page_range_label(), "pages 9-14")

    def test_section_pdf_cache_key_changes_when_section_changes(self):
        section_one = self._section("section-1", 1, 8)
        section_two = self._section("section-2", 9, 14)

        self.assertNotEqual(
            section_pdf_cache_key("notes.pdf", section_one),
            section_pdf_cache_key("notes.pdf", section_two),
        )

    def test_sanitizer_removes_nested_source_metadata(self):
        text = "[Section: [Source: Page 1], Page 1] chunk_id: abc Gradient descent"

        self.assertEqual(sanitize_visible_text(text), "Gradient descent")

    def test_cropped_pdf_returns_valid_pdf_bytes(self):
        source_pdf = self._sample_pdf_bytes(page_count=3)

        cropped = create_section_pdf(source_pdf, 2, 3, "quality-test")

        self.assertTrue(cropped.startswith(b"%PDF"))
        self.assertLess(len(cropped), len(source_pdf))

    def test_invalid_page_range_does_not_crash(self):
        source_pdf = self._sample_pdf_bytes(page_count=2)

        cropped = create_section_pdf(source_pdf, 99, 100, "invalid-range")

        self.assertEqual(cropped, source_pdf)

    def test_render_pdf_pages_to_images_returns_images_for_valid_pdf(self):
        source_pdf = self._sample_pdf_bytes(page_count=2)

        images = render_pdf_pages_to_images(source_pdf, dpi=72)

        self.assertEqual(len(images), 2)
        self.assertTrue(all(image.startswith(b"\x89PNG") for image in images))

    def test_render_pdf_pages_to_images_returns_empty_for_invalid_pdf(self):
        self.assertEqual(render_pdf_pages_to_images(b"not a pdf"), [])

    def test_rendered_pdf_section_changes_when_section_changes(self):
        source_pdf = self._sample_pdf_bytes(page_count=3)
        first = create_section_pdf(source_pdf, 1, 1, "section-one")
        second = create_section_pdf(source_pdf, 2, 3, "section-two")

        first_images = render_pdf_pages_to_images(first, dpi=72)
        second_images = render_pdf_pages_to_images(second, dpi=72)

        self.assertEqual(len(first_images), 1)
        self.assertEqual(len(second_images), 2)

    def test_timer_state_calculation(self):
        self.assertEqual(calculate_elapsed_seconds(30, None, now=100), 30)
        self.assertEqual(calculate_elapsed_seconds(30, 90, now=100), 40)
        self.assertEqual(calculate_elapsed_seconds(30, 110, now=100), 30)

    @staticmethod
    def _sample_pdf_bytes(page_count: int) -> bytes:
        from pypdf import PdfWriter

        writer = PdfWriter()
        for _page in range(page_count):
            writer.add_blank_page(width=72, height=72)
        output = BytesIO()
        writer.write(output)
        return output.getvalue()

    @staticmethod
    def _section(section_id: str, page_start: int, page_end: int) -> StudySection:
        return StudySection(
            section_id=section_id,
            title=section_id,
            page_start=page_start,
            page_end=page_end,
            summary="summary",
            key_concepts=["concept"],
            estimated_minutes=10,
            difficulty="Easy",
            chunk_ids=["chunk"],
            content_preview="preview",
            source_id="notes.pdf",
        )


if __name__ == "__main__":
    unittest.main()
