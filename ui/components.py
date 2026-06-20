from __future__ import annotations

import html

import streamlit as st

from services.study_service import StudySection
from translations import SUPPORTED_LANGUAGES, t
from ui.navigation import NAV_ITEMS, NAV_TRANSLATION_KEYS
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
        header_columns = st.columns([3.5, 1.25])
        with header_columns[0]:
            st.markdown(f'<div class="brand">{html.escape(t("app_title"))}</div>', unsafe_allow_html=True)
        with header_columns[1]:
            st.selectbox(
                t("language"),
                options=list(SUPPORTED_LANGUAGES),
                format_func=lambda language: SUPPORTED_LANGUAGES[language],
                key="language",
            )

        nav_columns = st.columns([1, 1.15, 1.15, 1, 1, 1])
        for index, label in enumerate(NAV_ITEMS):
            with nav_columns[index]:
                display_label = t(NAV_TRANSLATION_KEYS[label])
                if st.session_state.current_page == label:
                    st.markdown(f'<div class="nav-active">{html.escape(display_label)}</div>', unsafe_allow_html=True)
                elif st.button(display_label, key=f"nav-{label}", use_container_width=True):
                    st.session_state.current_page = label
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_status_bar() -> None:
    if has_pdf():
        status = (
            f"<span><strong>{html.escape(t('current_pdf'))}:</strong> {html.escape(st.session_state.pdf_name)}</span>"
            f"<span><strong>{html.escape(t('pages'))}:</strong> {len(st.session_state.pages)}</span>"
            f"<span><strong>{html.escape(t('sessions'))}:</strong> {len(st.session_state.sections)}</span>"
            f"<span><strong>{html.escape(t('progress'))}:</strong> {overall_progress():.0f}%</span>"
        )
    else:
        status = t("no_pdf_loaded")
    st.markdown(f'<div class="status-bar">{status}</div>', unsafe_allow_html=True)
    if has_pdf() and st.session_state.current_page != "Upload":
        action_cols = st.columns([1, 5])
        with action_cols[0]:
            if st.button(t("change_pdf"), key="change-pdf", use_container_width=True):
                st.session_state.current_page = "Upload"
                source = "folder" if st.session_state.uploaded_folder_files else "file"
                st.session_state.upload_source = source
                st.rerun()


def render_upload_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{html.escape(t("upload_hero_title"))}</h1>
            <p>{html.escape(t("upload_hero_body"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
