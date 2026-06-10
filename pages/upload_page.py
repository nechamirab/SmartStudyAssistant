from __future__ import annotations

import streamlit as st

from services.pdf_service import PdfExtractionError
from services.study_service import StudyService
from ui.components import render_upload_hero
from ui.workflow import extract_pdf, generate_study_plan_from_pending


def render_upload() -> None:
    render_upload_hero()
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        signature = uploaded_file_signature(uploaded_file)
        if st.session_state.processed_upload_signature != signature:
            try:
                with st.spinner("Extracting text from your PDF..."):
                    extract_pdf(uploaded_file)
                st.session_state.processed_upload_signature = signature
                st.success(st.session_state.upload_message)
            except PdfExtractionError as exc:
                st.error(f"PDF extraction failed: {exc}")
            except Exception as exc:
                st.error(f"Could not process PDF safely: {exc}")

    if uploaded_file is not None and st.button("Reprocess PDF"):
        try:
            with st.spinner("Extracting text from your PDF..."):
                extract_pdf(uploaded_file)
            st.session_state.processed_upload_signature = uploaded_file_signature(uploaded_file)
            st.success(st.session_state.upload_message)
        except PdfExtractionError as exc:
            st.error(f"PDF extraction failed: {exc}")
        except Exception as exc:
            st.error(f"Could not process PDF safely: {exc}")

    if st.session_state.pending_pages:
        readable_pages = StudyService.readable_page_count(st.session_state.pending_pages)
        suggested = st.session_state.suggested_session_count or StudyService.suggest_session_count(
            st.session_state.pending_pages
        )
        max_sessions = 15
        selected = st.number_input(
            "Number of Study Sessions",
            min_value=3,
            max_value=max_sessions,
            value=max(3, min(max_sessions, int(st.session_state.selected_session_count or suggested or 5))),
            help="Suggested based on PDF length. You can change it.",
        )
        st.session_state.selected_session_count = int(selected)
        draft_sections = StudyService().generate_study_plan_for_sessions(
            st.session_state.pending_pages,
            st.session_state.selected_session_count,
        )
        st.session_state.pending_sections = draft_sections
        estimated = sum(section.estimated_minutes for section in draft_sections)
        cols = st.columns(3)
        cols[0].metric("Pages processed", len(st.session_state.pending_pages))
        cols[1].metric("Suggested sessions", suggested)
        cols[2].metric("Estimated study time", f"{estimated} min")
        st.caption(
            f"Suggested sessions: {suggested}. Your study plan will use {len(draft_sections)} study sessions."
        )
        if st.button("Generate Study Plan", type="primary"):
            try:
                generate_study_plan_from_pending()
                st.success(st.session_state.upload_message)
                st.session_state.current_page = "Study Plan"
                st.rerun()
            except PdfExtractionError as exc:
                st.error(str(exc))


def uploaded_file_signature(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"
