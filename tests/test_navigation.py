from __future__ import annotations

import unittest

from pages.ai_tutor_page import render_ai_tutor
from pages.dashboard_page import render_dashboard
from pages.final_exam_page import render_final_exam
from pages.study_mode_page import render_study_mode, should_show_next_section_button
from pages.study_plan_page import render_study_plan
from pages.upload_page import render_upload
from ui.navigation import DEFAULT_CURRENT_PAGE, NAV_ITEMS, normalize_current_page
from ui.streamlit_app import PAGE_RENDERERS


class NavigationTests(unittest.TestCase):
    def test_navigation_steps_exist(self):
        self.assertEqual(
            NAV_ITEMS,
            ("Upload", "Study Plan", "Study Mode", "AI Tutor", "Final Exam", "Dashboard"),
        )

    def test_removed_navigation_steps_are_not_present(self):
        removed_steps = {"RAG Check", "Quizzes", "OCR", "Results", "About"}
        self.assertTrue(removed_steps.isdisjoint(NAV_ITEMS))

    def test_current_page_defaults_to_upload(self):
        self.assertEqual(DEFAULT_CURRENT_PAGE, "Upload")
        self.assertEqual(normalize_current_page(None), "Upload")
        self.assertEqual(normalize_current_page("Legacy Page"), "Upload")

    def test_page_renderers_cover_every_navigation_step(self):
        self.assertEqual(set(PAGE_RENDERERS), set(NAV_ITEMS))
        self.assertIs(PAGE_RENDERERS["Upload"], render_upload)
        self.assertIs(PAGE_RENDERERS["Study Plan"], render_study_plan)
        self.assertIs(PAGE_RENDERERS["Study Mode"], render_study_mode)
        self.assertIs(PAGE_RENDERERS["AI Tutor"], render_ai_tutor)
        self.assertIs(PAGE_RENDERERS["Final Exam"], render_final_exam)
        self.assertIs(PAGE_RENDERERS["Dashboard"], render_dashboard)

    def test_next_section_button_hidden_on_last_session(self):
        self.assertTrue(should_show_next_section_button(0, 2))
        self.assertFalse(should_show_next_section_button(1, 2))
        self.assertFalse(should_show_next_section_button(0, 0))


if __name__ == "__main__":
    unittest.main()
