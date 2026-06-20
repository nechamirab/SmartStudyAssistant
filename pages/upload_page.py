from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from services.auth_service import AuthService
from services.database_service import DatabaseService
from services.pdf_service import PdfExtractionError
from services.study_service import StudyService
from translations import current_language, t
from ui.components import render_upload_hero
from ui.workflow import (
    extract_pdf,
    generate_study_plan_from_pending,
    load_saved_study_session,
    pending_study_plan_signature,
)


@dataclass(frozen=True)
class StoredUploadedPdf:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def render_upload() -> None:
    render_upload_hero()
    render_saved_sessions()
    st.subheader(t("upload_pdf"))
    source_options = ["file", "folder"]
    saved_source = st.session_state.get("upload_source", "file")
    if saved_source not in source_options:
        saved_source = "file"
    if st.session_state.get("uploaded_folder_files") and saved_source == "folder":
        st.session_state.upload_source_choice = "folder"
    elif st.session_state.get("upload_source_choice") not in source_options:
        st.session_state.upload_source_choice = saved_source

    source = st.radio(
        t("pdf_source"),
        source_options,
        format_func=lambda value: t("upload_file") if value == "file" else t("upload_folder"),
        horizontal=True,
        key="upload_source_choice",
    )
    st.session_state.upload_source = source

    if source == "file":
        render_file_upload()
    else:
        render_folder_upload()

    render_pending_study_plan()


def render_saved_sessions() -> None:
    user = AuthService().current_user()
    if not user:
        return
    try:
        sessions = DatabaseService().list_study_sessions(int(user["id"]))
    except Exception as exc:
        st.warning(f"Could not load saved sessions: {exc}")
        return

    st.subheader("Continue Previous Study Session")
    if not sessions:
        st.caption("No saved sessions yet. Upload a PDF to create one.")
        return

    for item in sessions[:5]:
        with st.container(border=True):
            cols = st.columns([0.45, 0.2, 0.2, 0.15])
            cols[0].markdown(f"**{item['filename']}**")
            cols[0].caption(item["title"])
            cols[1].metric("Progress", f"{item['progress_percent']}%")
            cols[2].caption(f"Updated: {item['updated_at']}")
            if cols[3].button("Continue", key=f"continue-session-{item['id']}"):
                if load_saved_study_session(int(item["id"])):
                    st.session_state.current_page = "Study Mode"
                    st.rerun()


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
    st.session_state.upload_source = "folder"
    uploaded_files = st.file_uploader(
        t("choose_pdf_folder"),
        type=["pdf"],
        accept_multiple_files="directory",
        key="folder_pdf_uploader",
    )
    if uploaded_files:
        save_uploaded_folder_files(uploaded_files)

    saved_files = st.session_state.uploaded_folder_files
    if not saved_files:
        st.info(t("folder_upload_help"))
        return

    labels = [file["name"] for file in saved_files]
    if st.session_state.selected_folder_pdf not in labels:
        st.session_state.selected_folder_pdf = labels[0]

    selected_label = st.selectbox(
        t("choose_pdf_from_folder"),
        labels,
        index=labels.index(st.session_state.selected_folder_pdf),
        key="selected_folder_pdf",
    )
    selected_file = stored_uploaded_pdf(saved_files[labels.index(selected_label)])
    st.caption(t("pdfs_uploaded_from_folder", count=len(saved_files)))

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
            st.session_state.pending_plan_signature = ""
            st.session_state.pending_pdf_bytes = b""
            st.session_state.pending_pdf_name = ""
            st.session_state.processed_upload_signature = ""
            source = "folder" if st.session_state.uploaded_folder_files else "file"
            st.session_state.upload_source = source
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
        signature = pending_study_plan_signature(
            st.session_state.pending_pdf_name,
            st.session_state.pending_pages,
            st.session_state.selected_session_count,
            current_language(),
        )
        if st.session_state.pending_sections and st.session_state.pending_plan_signature == signature:
            draft_sections = st.session_state.pending_sections
        else:
            draft_sections = StudyService().generate_study_plan_for_sessions(
                st.session_state.pending_pages,
                st.session_state.selected_session_count,
                language=current_language(),
            )
            st.session_state.pending_sections = draft_sections
            st.session_state.pending_plan_signature = signature
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


def save_uploaded_folder_files(uploaded_files) -> None:
    saved = [
        {
            "name": uploaded_file.name,
            "data": uploaded_file.getvalue(),
        }
        for uploaded_file in sorted(uploaded_files, key=lambda file: file.name.lower())
    ]
    st.session_state.uploaded_folder_files = saved
    labels = [item["name"] for item in saved]
    if labels and st.session_state.selected_folder_pdf not in labels:
        st.session_state.selected_folder_pdf = labels[0]


def stored_uploaded_pdf(payload: dict[str, bytes | str]) -> StoredUploadedPdf:
    return StoredUploadedPdf(
        name=str(payload["name"]),
        data=bytes(payload["data"]),
    )
