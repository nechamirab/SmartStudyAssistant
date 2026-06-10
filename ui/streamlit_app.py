from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pages.ai_tutor_page import render_ai_tutor
from pages.dashboard_page import render_dashboard
from pages.final_exam_page import render_final_exam
from pages.study_mode_page import render_study_mode
from pages.study_plan_page import render_study_plan
from pages.upload_page import render_upload
from ui.components import render_status_bar, render_top_nav
from ui.navigation import NAV_ITEMS
from ui.state import init_state
from ui.styles import inject_custom_css


PAGE_RENDERERS = {
    "Upload": render_upload,
    "Study Plan": render_study_plan,
    "Study Mode": render_study_mode,
    "AI Tutor": render_ai_tutor,
    "Final Exam": render_final_exam,
    "Dashboard": render_dashboard,
}


def main() -> None:
    st.set_page_config(page_title="Smart Study Assistant", page_icon=":books:", layout="wide")
    inject_custom_css()
    init_state()
    render_top_nav()
    render_status_bar()

    PAGE_RENDERERS[st.session_state.current_page]()


if __name__ == "__main__":
    missing_pages = set(NAV_ITEMS) - set(PAGE_RENDERERS)
    if missing_pages:
        raise RuntimeError(f"Missing page renderers: {sorted(missing_pages)}")
    main()
