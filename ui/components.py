from __future__ import annotations

import html

import streamlit as st

from services.study_service import StudySection
from ui.navigation import NAV_ITEMS
from ui.state import has_pdf, overall_progress


def card(title: str, body: str, extra: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{html.escape(title)}</div>
            <div>{body}</div>
            {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


def roadmap_card(section: StudySection, body: str) -> None:
    st.markdown(
        f"""
        <div class="card roadmap-card">
            <div class="card-title"><span class="roadmap-index">{section.section_number}</span>{html.escape(section.title)}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "primary") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def render_top_nav() -> None:
    with st.container(border=True):
        st.markdown('<div class="top-nav">', unsafe_allow_html=True)
        columns = st.columns([2.25, 1, 1.15, 1.15, 1, 1, 1])
        with columns[0]:
            st.markdown('<div class="brand">Smart Study Assistant</div>', unsafe_allow_html=True)
        for index, label in enumerate(NAV_ITEMS, start=1):
            with columns[index]:
                if st.session_state.current_page == label:
                    st.markdown(f'<div class="nav-active">{html.escape(label)}</div>', unsafe_allow_html=True)
                elif st.button(label, key=f"nav-{label}", use_container_width=True):
                    st.session_state.current_page = label
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_status_bar() -> None:
    if has_pdf():
        status = (
            f"<span><strong>Current PDF:</strong> {html.escape(st.session_state.pdf_name)}</span>"
            f"<span><strong>Pages:</strong> {len(st.session_state.pages)}</span>"
            f"<span><strong>Sessions:</strong> {len(st.session_state.sections)}</span>"
            f"<span><strong>Progress:</strong> {overall_progress():.0f}%</span>"
        )
    else:
        status = "No PDF loaded yet - upload a document to begin."
    st.markdown(f'<div class="status-bar">{status}</div>', unsafe_allow_html=True)


def render_upload_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Turn PDFs into guided study sessions, quizzes, and exam practice.</h1>
            <p>Upload course material, study section by section, and ask the AI Tutor for help.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
