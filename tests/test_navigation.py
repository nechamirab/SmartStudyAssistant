from __future__ import annotations

import unittest
from pathlib import Path

from ui.navigation import DEFAULT_CURRENT_PAGE, NAV_ITEMS, normalize_current_page


class NavigationTests(unittest.TestCase):
    def test_navigation_labels_are_exact(self):
        self.assertEqual(
            NAV_ITEMS,
            ("Upload", "Study Plan", "Study Mode", "AI Tutor", "Final Exam", "Dashboard"),
        )

    def test_removed_navigation_labels_are_not_present(self):
        self.assertNotIn("RAG Check", NAV_ITEMS)
        self.assertNotIn("Ask AI", NAV_ITEMS)
        self.assertNotIn("Quizzes", NAV_ITEMS)
        self.assertNotIn("OCR", NAV_ITEMS)
        self.assertNotIn("Results", NAV_ITEMS)
        self.assertNotIn("About", NAV_ITEMS)

    def test_current_page_defaults_to_upload(self):
        self.assertEqual(DEFAULT_CURRENT_PAGE, "Upload")
        self.assertEqual(normalize_current_page(None), "Upload")
        self.assertEqual(normalize_current_page("RAG Check"), "Upload")

    def test_streamlit_app_uses_custom_navigation_contract(self):
        source = Path("ui/streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn("inject_custom_css()", source)
        self.assertIn("render_top_nav()", source)
        self.assertIn("st.session_state.current_page", source)
        self.assertIn('PAGE_RENDERERS[st.session_state.current_page]()', source)
        self.assertNotIn("st.tabs(", source)
        self.assertNotIn("with st.sidebar:", source)
        self.assertNotIn('"Ask AI": render_ask_ai', source)
        self.assertIn('"AI Tutor": render_ask_ai', source)


if __name__ == "__main__":
    unittest.main()
