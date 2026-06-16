from __future__ import annotations

import streamlit as st

from services.pdf_service import PdfExtractionError
from services.study_service import StudyService
from translations import current_language, t
from ui.components import render_upload_hero
from ui.workflow import extract_pdf, generate_study_plan_from_pending


def render_upload() -> None:
    render_upload_hero()
    st.subheader(t("upload_pdf"))
    source_options = ["file", "folder"]
    source = st.radio(
        t("pdf_source"),
        source_options,
        format_func=lambda value: t("upload_file") if value == "file" else t("upload_folder"),
        horizontal=True,
    )

    if source == "file":
        render_file_upload()
    else:
        render_folder_upload()

    render_pending_study_plan()


def render_file_upload() -> None:
    uploaded_file = st.file_uploader(t("choose_pdf"), type=["pdf"])

    if uploaded_file is not None:
        signature = uploaded_file_signature(uploaded_file)
        if st.session_state.processed_upload_signature != signature:
            try:
                with st.spinner(t("extracting_pdf")):
                    extract_pdf(uploaded_file)
                st.session_state.processed_upload_signature = signature
                st.success(st.session_state.upload_message)
            except PdfExtractionError as exc:
                st.error(f"{t('pdf_extraction_failed')}: {exc}")
            except Exception as exc:
                st.error(f"{t('pdf_process_failed')}: {exc}")

    if uploaded_file is not None and st.button(t("reprocess_pdf")):
        try:
            with st.spinner(t("extracting_pdf")):
                extract_pdf(uploaded_file)
            st.session_state.processed_upload_signature = uploaded_file_signature(uploaded_file)
            st.success(st.session_state.upload_message)
        except PdfExtractionError as exc:
            st.error(f"{t('pdf_extraction_failed')}: {exc}")
        except Exception as exc:
            st.error(f"{t('pdf_process_failed')}: {exc}")


def render_folder_upload() -> None:
    uploaded_files = st.file_uploader(
        t("choose_pdf_folder"),
        type=["pdf"],
        accept_multiple_files="directory",
    )
    if not uploaded_files:
        st.info(t("folder_upload_help"))
        return

    uploaded_files = sorted(uploaded_files, key=lambda file: file.name.lower())
    labels = [file.name for file in uploaded_files]
    selected_label = st.selectbox(t("choose_pdf_from_folder"), labels)
    selected_file = uploaded_files[labels.index(selected_label)]
    st.caption(t("pdfs_uploaded_from_folder", count=len(uploaded_files)))

    signature = uploaded_file_signature(selected_file)
    button_label = (
        t("reprocess_selected_pdf")
        if st.session_state.processed_upload_signature == signature
        else t("process_selected_pdf")
    )
    if st.button(button_label, type="primary"):
        try:
            with st.spinner(t("extracting_pdf")):
                extract_pdf(selected_file)
            st.session_state.processed_upload_signature = signature
            st.success(st.session_state.upload_message)
        except PdfExtractionError as exc:
            st.error(f"{t('pdf_extraction_failed')}: {exc}")
        except Exception as exc:
            st.error(f"{t('pdf_process_failed')}: {exc}")


def render_pending_study_plan() -> None:
    if st.session_state.pending_pages:
        if st.button(t("choose_another_pdf")):
            st.session_state.pending_pages = []
            st.session_state.pending_sections = []
            st.session_state.pending_pdf_bytes = b""
            st.session_state.pending_pdf_name = ""
            st.session_state.processed_upload_signature = ""
            st.rerun()

        readable_pages = StudyService.readable_page_count(st.session_state.pending_pages)
        suggested = st.session_state.suggested_session_count or StudyService.suggest_session_count(
            st.session_state.pending_pages
        )
        max_sessions = 15
        selected = st.number_input(
            t("number_of_study_sessions"),
            min_value=3,
            max_value=max_sessions,
            value=max(3, min(max_sessions, int(st.session_state.selected_session_count or suggested or 5))),
            help=t("session_count_help"),
        )
        st.session_state.selected_session_count = int(selected)
        draft_sections = StudyService().generate_study_plan_for_sessions(
            st.session_state.pending_pages,
            st.session_state.selected_session_count,
            language=current_language(),
        )
        st.session_state.pending_sections = draft_sections
        estimated = sum(section.estimated_minutes for section in draft_sections)
        cols = st.columns(3)
        cols[0].metric(t("pages_processed"), len(st.session_state.pending_pages))
        cols[1].metric(t("suggested_sessions"), suggested)
        cols[2].metric(t("estimated_study_time"), f"{estimated} {t('minutes')}")
        st.caption(
            t("study_plan_session_caption", suggested=suggested, count=len(draft_sections))
        )
        if st.button(t("generate_study_plan"), type="primary"):
            try:
                generate_study_plan_from_pending()
                st.success(st.session_state.upload_message)
                st.session_state.current_page = "Study Plan"
                st.rerun()
            except PdfExtractionError as exc:
                st.error(str(exc))


def uploaded_file_signature(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"
