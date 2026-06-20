from __future__ import annotations

from typing import Any

import streamlit as st

from services.persistence_service import PersistenceService
from services.progress_service import ProgressService
from services.section_state_service import SectionStateService
from services.study_service import StudySection, StudyService
from translations import DEFAULT_LANGUAGE, normalize_language, t
from ui.navigation import DEFAULT_CURRENT_PAGE, normalize_current_page


def init_state() -> None:
    defaults = {
        "pdf_bytes": b"",
        "pdf_name": "",
        "pages": [],
        "pending_pages": [],
        "pending_sections": [],
        "pending_plan_signature": "",
        "pending_pdf_bytes": b"",
        "pending_pdf_name": "",
        "upload_source": "file",
        "upload_source_choice": "file",
        "uploaded_folder_files": [],
        "selected_folder_pdf": "",
        "processed_upload_signature": "",
        "suggested_session_count": 0,
        "selected_session_count": 0,
        "sections": [],
        "current_section_index": 0,
        "upload_message": "",
        "section_states": {},
        "ai_tutor_history": [],
        "final_exam": None,
        "final_exam_answers": {},
        "final_exam_result": None,
        "current_db_document_id": None,
        "current_db_session_id": None,
        "db_status_message": "",
        "active_auth_user_id": None,
        "weak_topic_review": "",
        "current_page": DEFAULT_CURRENT_PAGE,
        "language": DEFAULT_LANGUAGE,
        "progress": ProgressService.default_state(),
        "persistence_loaded": False,
        "sqlite_autoload_user_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    enforce_authenticated_user_isolation()
    restore_saved_state()
    restore_latest_sqlite_session()
    st.session_state.progress = ProgressService.load(st.session_state.progress)
    st.session_state.current_page = normalize_current_page(st.session_state.get("current_page"))
    st.session_state.language = normalize_language(st.session_state.get("language"))
    st.session_state.sections = normalize_study_sections(st.session_state.sections)
    st.session_state.pending_sections = normalize_study_sections(st.session_state.pending_sections)
    st.session_state.section_states = SectionStateService.ensure_states(
        st.session_state.section_states,
        [section.section_number for section in st.session_state.sections],
    )


def restore_saved_state() -> None:
    if st.session_state.persistence_loaded:
        return
    st.session_state.persistence_loaded = True
    if st.session_state.get("auth_user"):
        # Logged-in users should resume through SQLite saved sessions only.
        # Loading the legacy JSON cache here would leak one user's active PDF
        # into another user's browser session.
        return
    if st.session_state.sections or st.session_state.pages:
        return

    payload = PersistenceService.load()
    if not payload:
        return

    st.session_state.pdf_name = str(payload.get("pdf", {}).get("name", "") or "")
    st.session_state.pages = PersistenceService.pages_from_payload(payload)
    st.session_state.sections = PersistenceService.sections_from_payload(payload)
    st.session_state.current_section_index = int(payload.get("current_section_index", 0) or 0)
    st.session_state.progress = ProgressService.load(payload.get("progress", {}))
    st.session_state.section_states = payload.get("section_states", {})
    st.session_state.final_exam = payload.get("final_exam")
    st.session_state.final_exam_answers = dict(payload.get("final_exam_answers", {}) or {})
    st.session_state.final_exam_result = payload.get("final_exam_result")


def enforce_authenticated_user_isolation() -> None:
    auth_user = st.session_state.get("auth_user")
    current_user_id = None
    if isinstance(auth_user, dict) and auth_user.get("id"):
        current_user_id = int(auth_user["id"])

    if st.session_state.get("active_auth_user_id") == current_user_id:
        return

    reset_active_study_state()
    st.session_state.active_auth_user_id = current_user_id
    st.session_state.persistence_loaded = False


def reset_active_study_state() -> None:
    st.session_state.pdf_bytes = b""
    st.session_state.pdf_name = ""
    st.session_state.pages = []
    st.session_state.pending_pages = []
    st.session_state.pending_sections = []
    st.session_state.pending_plan_signature = ""
    st.session_state.pending_pdf_bytes = b""
    st.session_state.pending_pdf_name = ""
    st.session_state.uploaded_folder_files = []
    st.session_state.selected_folder_pdf = ""
    st.session_state.processed_upload_signature = ""
    st.session_state.suggested_session_count = 0
    st.session_state.selected_session_count = 0
    st.session_state.sections = []
    st.session_state.current_section_index = 0
    st.session_state.upload_message = ""
    st.session_state.section_states = {}
    st.session_state.ai_tutor_history = []
    st.session_state.final_exam = None
    st.session_state.final_exam_answers = {}
    st.session_state.final_exam_result = None
    st.session_state.current_db_document_id = None
    st.session_state.current_db_session_id = None
    st.session_state.db_status_message = ""
    st.session_state.weak_topic_review = ""
    st.session_state.progress = ProgressService.default_state()
    st.session_state.current_page = DEFAULT_CURRENT_PAGE
    st.session_state.sqlite_autoload_user_id = None


def restore_latest_sqlite_session() -> None:
    auth_user = st.session_state.get("auth_user")
    if not isinstance(auth_user, dict) or not auth_user.get("id"):
        return
    user_id = int(auth_user["id"])
    if st.session_state.get("sqlite_autoload_user_id") == user_id:
        return
    st.session_state.sqlite_autoload_user_id = user_id
    if st.session_state.pages or st.session_state.sections or st.session_state.current_db_session_id:
        return

    try:
        from services.database_service import DatabaseService

        database = DatabaseService()
        sessions = database.list_study_sessions(user_id)
        if not sessions:
            return
        payload = database.load_study_session(user_id, int(sessions[0]["id"]))
        if not payload:
            return
        apply_sqlite_session_payload(payload, status_message="Latest saved session restored.")
    except Exception as exc:
        st.session_state.db_status_message = f"SQLite session restore failed: {exc}"


def apply_sqlite_session_payload(payload: dict[str, Any], status_message: str = "Saved session loaded.") -> None:
    session = payload["session"]
    st.session_state.pdf_bytes = payload.get("pdf_bytes") or b""
    st.session_state.pdf_name = session.get("filename", "")
    st.session_state.pages = payload["pages"]
    st.session_state.sections = payload["sections"]
    st.session_state.section_states = payload["section_states"]
    st.session_state.progress = payload["progress"]
    st.session_state.final_exam = payload["final_exam"]
    st.session_state.final_exam_answers = payload["final_exam_answers"]
    st.session_state.final_exam_result = payload["final_exam_result"]
    st.session_state.current_section_index = int(session.get("current_section_index", 0) or 0)
    st.session_state.current_db_document_id = int(session["document_id"])
    st.session_state.current_db_session_id = int(session["id"])
    st.session_state.upload_message = f"Loaded saved session for {st.session_state.pdf_name}."
    st.session_state.db_status_message = status_message


def normalize_study_section(section: Any) -> StudySection:
    if isinstance(section, StudySection):
        return section

    if isinstance(section, dict):
        data = section
    elif hasattr(section, "__dict__"):
        data = vars(section)
    else:
        data = {}

    return StudySection(
        section_number=int(data.get("section_number", 0)),
        title=str(data.get("title", "")),
        start_page=int(data.get("start_page", 0)),
        end_page=int(data.get("end_page", 0)),
        estimated_minutes=int(data.get("estimated_minutes", 0)),
        difficulty=str(data.get("difficulty", "Easy")),
        summary=str(data.get("summary", "")),
        learning_objectives=list(data.get("learning_objectives") or []),
        key_concepts=list(data.get("key_concepts") or []),
    )


def normalize_study_sections(sections: list[Any]) -> list[StudySection]:
    if not sections:
        return []
    return [normalize_study_section(section) for section in sections]


def has_pdf() -> bool:
    return bool(st.session_state.pages and st.session_state.sections)


def current_section() -> StudySection | None:
    sections = st.session_state.sections
    if not sections:
        return None
    index = min(max(0, st.session_state.current_section_index), len(sections) - 1)
    st.session_state.current_section_index = index
    return sections[index]


def current_section_state() -> dict[str, Any]:
    section = current_section()
    if section is None:
        return SectionStateService.default_state()
    return SectionStateService.get_state(st.session_state.section_states, section.section_number)


def section_state(section: StudySection) -> dict[str, Any]:
    return SectionStateService.get_state(st.session_state.section_states, section.section_number)


def reset_section_outputs(section: StudySection | None = None) -> None:
    target = section or current_section()
    if target is None:
        return
    SectionStateService.reset_interaction_state(st.session_state.section_states, target.section_number)


def section_context(section: StudySection) -> str:
    return StudyService.section_text(st.session_state.pages, section)


def source_label(section: StudySection, page: int | None = None) -> str:
    if page is not None:
        return t("source_section_page", section=section.section_number, page=page)
    if section.start_page == section.end_page:
        return t("source_page", page=section.start_page)
    return t("source_pages", start=section.start_page, end=section.end_page)


def page_label(section: StudySection) -> str:
    if section.start_page == section.end_page:
        return t("page_label_one", page=section.start_page)
    return t("page_label_range", start=section.start_page, end=section.end_page)


def format_seconds(seconds: int) -> str:
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m" if st.session_state.get("language") != "he" else f"{hours}ש {minutes}ד"
    return f"{minutes}m {remainder}s" if st.session_state.get("language") != "he" else f"{minutes}ד {remainder}ש"


def overall_progress() -> float:
    total = len(st.session_state.sections)
    if total == 0:
        return 0.0
    return len(st.session_state.progress.completed_sections) / total * 100


def persist_current_state() -> None:
    # JSON persistence remains as a legacy/offline fallback. When a logged-in
    # user has an active SQLite session, the same runtime state is also saved
    # to the local database.
    payload = PersistenceService.build_payload(
        pdf_name=st.session_state.pdf_name,
        pages=st.session_state.pages,
        sections=st.session_state.sections,
        progress=ProgressService.load(st.session_state.progress),
        section_states=st.session_state.section_states,
        final_exam=st.session_state.final_exam,
        final_exam_answers=st.session_state.final_exam_answers,
        final_exam_result=st.session_state.final_exam_result,
        current_section_index=st.session_state.current_section_index,
    )
    PersistenceService.save(payload)
    persist_sqlite_state()


def persist_sqlite_state() -> None:
    session_id = st.session_state.get("current_db_session_id")
    if not session_id:
        return
    try:
        from services.auth_service import AuthService
        from services.database_service import DatabaseService

        user = AuthService().current_user()
        if not user:
            return
        DatabaseService().save_runtime_state(
            user_id=int(user["id"]),
            session_id=int(session_id),
            sections=st.session_state.sections,
            progress=ProgressService.load(st.session_state.progress),
            section_states=st.session_state.section_states,
            final_exam=st.session_state.final_exam,
            final_exam_answers=st.session_state.final_exam_answers,
            final_exam_result=st.session_state.final_exam_result,
            pdf_bytes=st.session_state.get("pdf_bytes", b""),
            current_section_index=st.session_state.current_section_index,
        )
    except Exception as exc:
        st.session_state.db_status_message = f"SQLite save failed: {exc}"
