from __future__ import annotations

import html
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_RETRIEVAL_SCORE, RETRIEVAL_TOP_K
from core.models import StudySection
from services.exam_service import ExamGenerationError, ExamRequest, FullExamService
from services.general_ai_service import answer_general_question
from services.pdf_render_service import render_pdf_pages_to_images
from services.pdf_section_service import create_section_pdf
from services.rag_service import PDFIndex, PDFRAGService, RAGPipelineError
from services.source_utils import (
    clean_section_title,
    format_page_range,
    format_source_label,
    normalize_bullets,
    sanitize_visible_text,
    source_metadata_from_ref,
)
import services.progress_service as progress_service_module
import services.study_service as study_service_module

progress_service_module = importlib.reload(progress_service_module)
study_service_module = importlib.reload(study_service_module)
ProgressService = progress_service_module.ProgressService
StudyService = study_service_module.StudyService
StudyServiceError = study_service_module.StudyServiceError


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --study-primary: #1e1b4b;
            --study-primary-hover: #312e81;
            --study-accent: #0891b2;
            --study-accent-soft: #cffafe;
            --study-ink: #111827;
            --study-muted: #5b6475;
            --study-border: #d7deea;
            --study-soft: #f4f7fb;
            --study-success: #047857;
            --study-warning: #a16207;
        }
        .stApp {
            background: linear-gradient(180deg, #eef4ff 0%, #ffffff 46%);
            color: var(--study-ink);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--study-ink);
        }
        h1 {
            font-size: 2.15rem;
            margin-bottom: 0.25rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6ecf2;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        div[data-testid="stTabs"] button {
            font-weight: 650;
            color: var(--study-muted);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--study-primary);
        }
        div[role="radiogroup"] label {
            border-radius: 999px;
            padding: 0.12rem 0.35rem;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            font-weight: 650;
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: var(--study-primary);
            border-color: var(--study-primary);
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background: var(--study-primary-hover);
            border-color: var(--study-primary-hover);
        }
        div[data-testid="stImage"] {
            background: #ffffff;
            border: 1px solid #e6ecf2;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
            margin-bottom: 0.85rem;
        }
        .app-note {
            border: 1px solid var(--study-border);
            border-left: 4px solid var(--study-primary);
            border-radius: 8px;
            background: #ffffff;
            padding: 13px 16px;
            margin-bottom: 16px;
            color: #243447;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .answer-box, .section-box, .study-card, .pdf-card {
            border: 1px solid var(--study-border);
            border-radius: 8px;
            background: #ffffff;
            padding: 16px;
            line-height: 1.58;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
        }
        .pdf-card {
            background: #f8fbff;
        }
        .pill {
            display: inline-block;
            border: 1px solid #a5f3fc;
            border-radius: 999px;
            padding: 3px 10px;
            margin: 2px 4px 2px 0;
            background: #ecfeff;
            font-size: 0.84rem;
            color: #155e75;
        }
        .source-badge {
            display: inline-block;
            border: 1px solid #a5f3fc;
            border-radius: 999px;
            padding: 2px 9px;
            margin: 2px 4px 6px 0;
            background: #ecfeff;
            color: #0e7490;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .muted {
            color: var(--study-muted);
            font-size: 0.92rem;
        }
        .section-kicker {
            color: var(--study-muted);
            font-size: 0.9rem;
            margin-top: -0.35rem;
            margin-bottom: 0.75rem;
        }
        .roadmap-card-title {
            font-size: 1.05rem;
            font-weight: 750;
            color: var(--study-primary);
            margin-bottom: 0.25rem;
        }
        .roadmap-meta {
            color: var(--study-muted);
            font-size: 0.9rem;
            margin-bottom: 0.65rem;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.78rem;
            font-weight: 750;
            margin-right: 6px;
            border: 1px solid transparent;
        }
        .badge-easy, .badge-completed {
            background: #dcfce7;
            border-color: #bbf7d0;
            color: var(--study-success);
        }
        .badge-medium, .badge-progress {
            background: #fef3c7;
            border-color: #fde68a;
            color: var(--study-warning);
        }
        .badge-hard {
            background: #ede9fe;
            border-color: #ddd6fe;
            color: #5b21b6;
        }
        .badge-not-started {
            background: #eef2f7;
            border-color: #dbe4ee;
            color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme() -> None:
    inject_custom_css()


def init_state() -> None:
    defaults = {
        "index": None,
        "active_page": "Upload",
        "sections": [],
        "current_section_index": 0,
        "started_sections": [],
        "last_answer": None,
        "last_sources": [],
        "section_explanation": None,
        "section_quiz": None,
        "section_quiz_grade": None,
        "final_exam": None,
        "final_exam_grade": None,
        "progress": None,
        "uploaded_pdf_files": {},
        "section_timer_started_at": {},
        "section_answer": None,
        "ai_chat_history": [],
        "exam_focus": {},
        "flashcards": {},
        "understanding_result": None,
        "mistake_review": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def rag_service() -> PDFRAGService:
    return PDFRAGService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=RETRIEVAL_TOP_K,
        min_score=MIN_RETRIEVAL_SCORE,
    )


def study_service() -> StudyService:
    return StudyService()


def progress_service() -> ProgressService:
    return ProgressService()


def current_index() -> PDFIndex | None:
    return st.session_state.get("index")


def current_sections() -> list[StudySection]:
    return st.session_state.get("sections") or []


def current_progress() -> dict:
    progress = st.session_state.get("progress") or {}
    sections = current_sections()
    if current_index() and (not progress or progress.get("document_name") != current_index().pdf_name):
        progress = progress_service().load_document(current_index().pdf_name, len(sections))
        st.session_state["progress"] = progress
    return progress


def require_index() -> PDFIndex | None:
    index = current_index()
    if not index:
        st.info("Upload and process a PDF first.")
    return index


def require_sections() -> list[StudySection]:
    sections = current_sections()
    if not sections:
        st.info("Generate a study plan first.")
    return sections


def selected_section() -> StudySection | None:
    sections = current_sections()
    if not sections:
        return None
    index = min(st.session_state.get("current_section_index", 0), len(sections) - 1)
    return sections[index]


def study_mode_tabs() -> list[str]:
    return ["Upload", "Study Plan", "Study Mode", "Ask AI", "Final Exam", "Dashboard"]


def set_active_page(page: str) -> None:
    labels = study_mode_tabs()
    selected = page if page in labels else labels[0]
    st.session_state["active_page"] = selected
    st.session_state["nav_page"] = selected


def active_page() -> str:
    labels = study_mode_tabs()
    page = st.session_state.get("active_page", labels[0])
    return page if page in labels else labels[0]


def section_pdf_cache_key(pdf_name: str | None, section: StudySection) -> str:
    return f"{pdf_name or section.source_id}:{section.section_id}:{section.page_start}-{section.page_end}"


def next_section_index(position: int, section_count: int) -> int:
    if section_count <= 0:
        return 0
    return min(max(0, int(position)) + 1, section_count - 1)


def reset_section_artifacts() -> None:
    st.session_state["section_explanation"] = None
    st.session_state["section_quiz"] = None
    st.session_state["section_quiz_grade"] = None
    st.session_state["section_answer"] = None
    st.session_state["understanding_result"] = None
    st.session_state["mistake_review"] = None


def pause_section_timer(section_id: str) -> None:
    timers = dict(st.session_state.get("section_timer_started_at", {}))
    started_at = timers.pop(section_id, None)
    if started_at:
        st.session_state["progress"] = progress_service().record_study_time(
            current_progress(),
            section_id,
            time.time() - float(started_at),
        )
    st.session_state["section_timer_started_at"] = timers


def remember_last_studied(section_id: str) -> None:
    progress = dict(current_progress())
    progress["last_studied_section"] = section_id
    st.session_state["progress"] = progress_service().save_document(progress)


def go_to_next_section(current_position: int, sections: list[StudySection]) -> None:
    if not sections or current_position >= len(sections) - 1:
        return
    current_section = sections[current_position]
    pause_section_timer(current_section.section_id)
    remember_last_studied(current_section.section_id)
    st.session_state["current_section_index"] = next_section_index(current_position, len(sections))
    reset_section_artifacts()


def section_progress_status(section: StudySection, progress: dict, current_section_id: str | None = None) -> str:
    completed = set(progress.get("completed_sections", []))
    if section.section_id in completed:
        return "Completed"
    if section.section_id == current_section_id or progress.get("last_studied_section") == section.section_id:
        return "In progress"
    return "Not started"


def badge_class(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    if normalized in {"easy", "completed"}:
        return "badge badge-easy"
    if normalized in {"medium", "in-progress"}:
        return "badge badge-medium"
    if normalized == "hard":
        return "badge badge-hard"
    return "badge badge-not-started"


def render_study_plan_card(section: StudySection, position: int, total: int, progress: dict) -> None:
    current = selected_section()
    status = section_progress_status(section, progress, current.section_id if current else None)
    title = clean_section_title(section.title, f"Section {position + 1}")
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="roadmap-card-title">Section {position + 1}: {html.escape(title)}</div>
            <div class="roadmap-meta">{html.escape(section.page_range_label())} · {section.estimated_minutes} min</div>
            <span class="{badge_class(section.difficulty)}">{html.escape(section.difficulty)}</span>
            <span class="{badge_class(status)}">{html.escape(status)}</span>
            """,
            unsafe_allow_html=True,
        )
        st.write(sanitize_visible_text(section.summary, remove_files=False))
        render_concepts(section.key_concepts[:8])
        if st.button(
            "Start session",
            key=f"study-plan-{section.section_id}",
            type="primary" if status != "Completed" else "secondary",
            use_container_width=True,
        ):
            current_section = selected_section()
            if current_section:
                pause_section_timer(current_section.section_id)
                remember_last_studied(current_section.section_id)
            st.session_state["current_section_index"] = position
            reset_section_artifacts()
            set_active_page("Study Mode")
            st.rerun()


def normalize_uploads(uploads: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], dict[str, bytes]]:
    normalized_uploads: list[tuple[str, bytes]] = []
    pdf_files: dict[str, bytes] = {}
    seen_names: dict[str, int] = {}
    for filename, pdf_bytes in uploads:
        safe_name = Path(filename).name
        seen_names[safe_name] = seen_names.get(safe_name, 0) + 1
        if seen_names[safe_name] > 1:
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            safe_name = f"{stem}-{seen_names[safe_name]}{suffix}"
        normalized_uploads.append((safe_name, pdf_bytes))
        pdf_files[safe_name] = pdf_bytes
    return normalized_uploads, pdf_files


def pdf_bytes_for_section(section: StudySection) -> tuple[str, bytes] | tuple[None, None]:
    pdf_files = st.session_state.get("uploaded_pdf_files") or {}
    if section.source_id in pdf_files:
        return section.source_id, pdf_files[section.source_id]
    if len(pdf_files) == 1:
        name, pdf_bytes = next(iter(pdf_files.items()))
        return name, pdf_bytes
    return None, None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {remaining}s"


def calculate_elapsed_seconds(saved_seconds: float, started_at: float | None, now: float | None = None) -> float:
    saved = max(0.0, float(saved_seconds or 0.0))
    if started_at:
        current = time.time() if now is None else float(now)
        return saved + max(0.0, current - float(started_at))
    return saved


def elapsed_section_seconds(progress: dict, section_id: str) -> float:
    saved = float(progress.get("section_time_seconds", {}).get(section_id, 0.0))
    started_at = st.session_state.get("section_timer_started_at", {}).get(section_id)
    return calculate_elapsed_seconds(saved, started_at)


def timer_running(section_id: str) -> bool:
    return bool(st.session_state.get("section_timer_started_at", {}).get(section_id))


def section_estimates(sections: list[StudySection]) -> dict[str, int]:
    return {section.section_id: section.estimated_minutes for section in sections}


def save_rag_evaluation(evaluation: dict) -> None:
    path = ROOT / ".cache" / "rag_evaluations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except json.JSONDecodeError:
        existing = []
    existing.append(evaluation)
    path.write_text(json.dumps(existing[-200:], indent=2), encoding="utf-8")


def cached_exam_focus(section: StudySection, index: PDFIndex) -> dict:
    focus_by_section = st.session_state.get("exam_focus", {})
    focus = focus_by_section.get(section.section_id)
    if not focus:
        focus = study_service().exam_focus(section, index)
        focus_by_section[section.section_id] = focus
        st.session_state["exam_focus"] = focus_by_section
    return focus


def cached_flashcards(section: StudySection, index: PDFIndex) -> list[dict]:
    cards_by_section = st.session_state.get("flashcards", {})
    cards = cards_by_section.get(section.section_id)
    if cards is None:
        cards = study_service().generate_flashcards(section, index)
        cards_by_section[section.section_id] = cards
        st.session_state["flashcards"] = cards_by_section
    return cards


def render_header() -> None:
    st.title("Smart Study Assistant")
    st.markdown(
        """
        <div class="app-note">
        <strong>Turn PDFs into guided study sessions, quizzes, and exam practice.</strong><br>
        Upload course material, learn section by section, and use the AI tutor for general study help.
        </div>
        """,
        unsafe_allow_html=True,
    )

    index = current_index()
    if not index:
        st.caption("No PDF indexed yet.")
        return

    summary = index.to_summary()
    progress = current_progress()
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("PDF", summary["pdf_name"])
    col_b.metric("Pages", summary["pages"])
    col_c.metric("Source passages", summary["chunks"])
    col_d.metric("Sections", len(current_sections()))
    col_e.metric("Progress", f"{progress.get('total_progress_percentage', 0):.1f}%")


def render_upload() -> None:
    st.subheader("Upload PDF")
    uploaded_files = st.file_uploader("Choose one or more PDFs", type=["pdf"], accept_multiple_files=True)
    ocr_mode = st.selectbox(
        "OCR mode",
        ["auto", "off", "force"],
        index=0,
        help="Auto uses normal PDF text first and OCRs pages with little or no extracted text.",
    )
    target_sections = st.slider("Study plan detail", min_value=3, max_value=12, value=6)
    uploads = [(uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files]

    if uploads:
        total_bytes = sum(len(pdf_bytes) for _filename, pdf_bytes in uploads)
        col_a, col_b = st.columns(2)
        col_a.metric("Selected files", len(uploads))
        col_b.metric("Total size", f"{total_bytes / 1024:.1f} KB")

    if st.button("Process PDF and Generate Study Plan", type="primary", disabled=not uploads):
        try:
            with st.spinner("Extracting text, chunking PDF, indexing sources, and creating study sections..."):
                normalized_uploads, pdf_files = normalize_uploads(uploads)
                index = rag_service().build_index_from_uploads(normalized_uploads, ocr_mode=ocr_mode)
                sections = study_service().create_study_plan(index, target_section_count=target_sections)
                progress = progress_service().load_document(index.pdf_name, len(sections))
                progress["section_count"] = len(sections)
                progress = progress_service().save_document(progress)
                st.session_state.update(
                    {
                        "index": index,
                        "sections": sections,
                        "progress": progress,
                        "current_section_index": 0,
                        "started_sections": [],
                        "last_answer": None,
                        "last_sources": [],
                        "section_explanation": None,
                        "section_quiz": None,
                        "section_quiz_grade": None,
                        "section_answer": None,
                        "ai_chat_history": [],
                        "final_exam": None,
                        "final_exam_grade": None,
                        "uploaded_pdf_files": pdf_files,
                        "section_timer_started_at": {},
                        "exam_focus": {},
                        "flashcards": {},
                        "understanding_result": None,
                        "mistake_review": None,
                    }
                )
            st.success(f"Created {len(sections)} study sections from {index.pdf_name}.")
        except (RAGPipelineError, StudyServiceError) as exc:
            st.error(str(exc))

    index = current_index()
    if index:
        extraction = index.extraction_summary()
        st.markdown("### Processing Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Pages processed", extraction["pages_processed"])
        col_b.metric("Normal text pages", extraction["pages_using_normal_text"])
        col_c.metric("OCR pages", extraction["pages_using_ocr"])
        col_d.metric("Characters", extraction["total_characters_extracted"])


def render_study_plan() -> None:
    st.subheader("Study Plan")
    index = require_index()
    sections = require_sections()
    if not index or not sections:
        st.markdown(
            "<div class='study-card'>Upload a PDF and generate a study plan to see your course roadmap here.</div>",
            unsafe_allow_html=True,
        )
        return

    progress = current_progress()
    completed = set(progress.get("completed_sections", []))
    total_minutes = sum(section.estimated_minutes for section in sections)
    st.markdown("Build a steady path through the uploaded material, one section at a time.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sections", len(sections))
    col_b.metric("Total estimated time", f"{total_minutes} min")
    col_c.metric("Completed", f"{len(completed)} / {len(sections)}")
    st.progress(progress.get("total_progress_percentage", 0) / 100)
    st.caption(f"Overall study progress: {progress.get('total_progress_percentage', 0):.1f}%")

    for position, section in enumerate(sections):
        render_study_plan_card(section, position, len(sections), progress)


def render_pdf_viewer(section: StudySection) -> None:
    pdf_name, pdf_bytes = pdf_bytes_for_section(section)
    st.markdown("### PDF Section")
    st.markdown(f"<div class='pdf-card'><strong>Now studying {format_page_range(section.page_start, section.page_end)}</strong></div>", unsafe_allow_html=True)
    if not pdf_bytes:
        st.info("The original PDF is not available in this session. Re-upload the PDF to use the split-screen viewer.")
        st.markdown("### Source Preview")
        escaped_preview = html.escape(sanitize_visible_text(section.content_preview, remove_files=False)).replace("\n", "<br>")
        st.markdown(f"<div class='section-box'>{escaped_preview}</div>", unsafe_allow_html=True)
        return

    cache_key = section_pdf_cache_key(pdf_name, section)
    section_pdf_bytes = create_section_pdf(pdf_bytes, section.page_start, section.page_end, cache_key)
    page_images = render_pdf_pages_to_images(section_pdf_bytes)
    if page_images:
        for page_number, image_bytes in enumerate(page_images, 1):
            st.image(image_bytes, use_container_width=True)
            if len(page_images) > 1:
                st.caption(f"Section page {page_number} of {len(page_images)}")
    else:
        st.warning("PDF preview could not be rendered. You can still study from the extracted text below.")

    section_filename = f"{Path(pdf_name or 'section.pdf').stem}-{section.section_id}.pdf"
    st.download_button(
        "Download section PDF",
        data=section_pdf_bytes or pdf_bytes,
        file_name=section_filename,
        mime="application/pdf",
        use_container_width=True,
    )
    with st.expander("Extracted text fallback"):
        escaped_preview = html.escape(sanitize_visible_text(section.content_preview, remove_files=False)).replace("\n", "<br>")
        st.markdown(f"<div class='section-box'>{escaped_preview}</div>", unsafe_allow_html=True)


def render_study_mode() -> None:
    st.subheader("Study Mode")
    index = require_index()
    sections = require_sections()
    if not index or not sections:
        return

    section = selected_section()
    if not section:
        return
    position = st.session_state.get("current_section_index", 0)
    progress = current_progress()
    completed = set(progress.get("completed_sections", []))

    pdf_col, assistant_col = st.columns([1.25, 1], gap="large")
    with pdf_col:
        render_pdf_viewer(section)

    with assistant_col:
        title = clean_section_title(section.title, f"Section {position + 1}")
        st.markdown(f"### {title}")
        status = "Completed" if section.section_id in completed else "In progress"
        st.markdown(
            f"<div class='section-kicker'>Section {position + 1} of {len(sections)} | "
            f"{section.page_range_label()} | {section.difficulty} | {status}</div>",
            unsafe_allow_html=True,
        )
        st.progress((position + 1) / len(sections))
        col_a, col_b = st.columns(2)
        actual_seconds = elapsed_section_seconds(progress, section.section_id)
        col_a.metric("Estimated time", f"{section.estimated_minutes} min")
        col_b.metric("Actual time", format_duration(actual_seconds))

        timer_cols = st.columns(3)
        if timer_cols[0].button("Start Timer", type="primary", disabled=timer_running(section.section_id), use_container_width=True):
            timers = dict(st.session_state.get("section_timer_started_at", {}))
            timers[section.section_id] = time.time()
            st.session_state["section_timer_started_at"] = timers
            st.rerun()
        if timer_cols[1].button("Pause", disabled=not timer_running(section.section_id), use_container_width=True):
            pause_section_timer(section.section_id)
            st.rerun()
        if timer_cols[2].button("Finish Section", use_container_width=True):
            pause_section_timer(section.section_id)
            st.session_state["progress"] = progress_service().mark_completed(current_progress(), section.section_id)
            st.success("Section finished.")

        st.markdown("<div class='study-card'>", unsafe_allow_html=True)
        if st.button("Generate Explanation", type="primary", use_container_width=True):
            try:
                st.session_state["section_explanation"] = study_service().explain_section(
                    section,
                    index,
                )
            except StudyServiceError as exc:
                st.error(str(exc))

        explanation = st.session_state.get("section_explanation")
        if explanation and explanation.get("section_id") == section.section_id:
            with st.expander("Explanation", expanded=True):
                st.write(sanitize_visible_text(explanation["explanation"], remove_files=False))
                st.markdown("**Key Definitions**")
                for item in explanation["definitions"]:
                    term = sanitize_visible_text(item.get("term", ""), remove_files=False)
                    definition = sanitize_visible_text(item.get("definition", ""), remove_files=False)
                    st.write(f"**{term}**: {definition}")
                st.markdown("**Important Points**")
                for point in normalize_bullets(explanation["important_points"], limit=5):
                    st.write(f"- {point}")
                st.markdown("**Example Questions**")
                for question in normalize_bullets(explanation["example_questions"], limit=3):
                    st.write(f"- {question}")
                render_source_refs(explanation.get("sources", []))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Quiz", expanded=False):
            if st.button("Generate Quiz", use_container_width=True):
                st.session_state["section_quiz"] = study_service().generate_section_quiz(section, index)
                st.session_state["section_quiz_grade"] = None
            render_active_section_quiz(section, index, namespace="study")

        with st.expander("Ask Question", expanded=False):
            question = st.text_input("Section question", key="section_question")
            if st.button("Ask Section", disabled=not question.strip(), use_container_width=True):
                result = rag_service().answer(
                    index,
                    question,
                    top_k=8,
                    metadata_filter={"section_id": section.section_id},
                )
                st.session_state["section_answer"] = result
            result = st.session_state.get("section_answer")
            if result and result.question:
                render_rag_answer(result)

        if st.button("Next Section", disabled=position >= len(sections) - 1, use_container_width=True):
            go_to_next_section(position, sections)
            st.rerun()


def render_quizzes() -> None:
    st.subheader("Quizzes")
    index = require_index()
    sections = require_sections()
    if not index or not sections:
        return

    section_titles = [section.title for section in sections]
    selected = st.selectbox("Choose section", section_titles)
    section = sections[section_titles.index(selected)]
    if st.button("Generate Mini Quiz", type="primary"):
        st.session_state["current_section_index"] = sections.index(section)
        st.session_state["section_quiz"] = study_service().generate_section_quiz(section, index)
        st.session_state["section_quiz_grade"] = None
    render_active_section_quiz(section, index, namespace="quizzes")


def render_active_section_quiz(section: StudySection, index: PDFIndex, namespace: str) -> None:
    quiz = st.session_state.get("section_quiz")
    if not quiz or quiz.get("section_id") != section.section_id:
        return

    st.markdown(f"### {quiz.get('title', 'Mini Quiz')}")
    answers: dict[str, str] = {}
    with st.form(f"quiz-form-{namespace}-{section.section_id}"):
        for question in quiz.get("questions", []):
            question_id = str(question["id"])
            st.markdown(f"**{question['id']}. {question['question']}**")
            if question["type"] in {"multiple_choice", "true_false"}:
                answers[question_id] = st.radio(
                    "Answer",
                    question.get("options", []),
                    key=f"quiz-{namespace}-{section.section_id}-{question_id}",
                    label_visibility="collapsed",
                )
            else:
                answers[question_id] = st.text_area(
                    "Answer",
                    key=f"quiz-{namespace}-{section.section_id}-{question_id}",
                    label_visibility="collapsed",
                )
            render_source_refs(question.get("source_references", []), compact=True)
        submitted = st.form_submit_button("Submit Quiz", type="primary")

    if submitted:
        grade = study_service().grade_quiz(quiz, answers)
        st.session_state["section_quiz_grade"] = grade
        progress = progress_service().record_quiz(current_progress(), section.section_id, grade)
        st.session_state["progress"] = progress

    grade = st.session_state.get("section_quiz_grade")
    if grade:
        render_grade(grade)
        if namespace != "study" and grade.get("weak_topics") and st.button(
            "Review My Mistakes",
            key=f"review-mistakes-{namespace}-{section.section_id}",
            use_container_width=True,
        ):
            st.session_state["mistake_review"] = study_service().generate_mistake_review(section, index, grade)

    review = st.session_state.get("mistake_review")
    if review and review.get("section_id") == section.section_id:
        st.markdown("### Review My Mistakes")
        st.write(review.get("review_lesson", ""))
        for item in review.get("wrong_questions", []):
            with st.expander(str(item.get("question", "Wrong question"))):
                st.write(f"Your answer: {item.get('student_answer') or 'No answer'}")
                st.write(f"Correct answer: {item.get('correct_answer')}")
                st.caption(item.get("explanation", ""))
                render_source_refs(item.get("source_references", []), compact=True)
        render_source_refs(review.get("sources", []))


def render_ask() -> None:
    st.subheader("Ask AI")
    st.caption("Ask general questions, request examples, or ask for help understanding a topic.")
    st.info("For answers grounded in your uploaded PDF, use Study Mode.")

    if st.button("Clear chat"):
        st.session_state["ai_chat_history"] = []
        st.rerun()

    history = st.session_state.get("ai_chat_history", [])
    for message in history:
        role = message.get("role", "assistant")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.write(message.get("content", ""))

    question = st.chat_input("Ask the AI tutor anything...")
    if question:
        history = list(st.session_state.get("ai_chat_history", []))
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_general_question(question, history[:-1])
                st.write(answer)
        history.append({"role": "assistant", "content": answer})
        st.session_state["ai_chat_history"] = history[-20:]


def render_final_exam() -> None:
    st.subheader("Final Exam")
    index = require_index()
    sections = require_sections()
    if not index or not sections:
        return

    completed_count = len(set(current_progress().get("completed_sections", [])))
    if completed_count < len(sections):
        st.info("You can generate the final exam anytime, but it is best after completing all sections.")

    col_a, col_b = st.columns(2)
    question_count = col_a.number_input("Number of questions", min_value=6, max_value=30, value=12)
    col_b.caption("Final exams use Groq AI when available, with a grounded fallback if the AI call fails.")

    if st.button("Generate Final Exam", type="primary"):
        try:
            with st.spinner("Generating a PDF-grounded practice exam..."):
                st.session_state["final_exam"] = FullExamService().generate_exam_with_fallback(
                    index,
                    ExamRequest(
                        number_of_questions=int(question_count),
                        question_types=["multiple_choice", "short_answer", "open_question"],
                        difficulty="mixed",
                        include_answer_key=True,
                    ),
                )
                st.session_state["final_exam_grade"] = None
        except (StudyServiceError, ExamGenerationError) as exc:
            st.error(str(exc))

    exam = st.session_state.get("final_exam")
    if not exam:
        return

    st.markdown(f"### {exam.get('title', 'Practice Final Exam')}")
    if exam.get("fallback_used"):
        st.warning(exam.get("fallback_note", "AI generation was unavailable, so a grounded fallback was used."))
    answers: dict[str, str] = {}
    with st.form("final-exam-form"):
        for question in exam.get("questions", []):
            question_id = str(question.get("id", len(answers) + 1))
            st.markdown(
                f"**{question_id}. {question.get('question', '')}**  \n"
                f"`{question.get('type', '')}` | `{question.get('difficulty', 'mixed')}`"
            )
            options = question.get("options") or []
            if question.get("type") == "multiple_choice" and options:
                answers[question_id] = st.radio(
                    "Answer",
                    options,
                    key=f"final-{question_id}",
                    label_visibility="collapsed",
                )
            elif question.get("type") == "true_false":
                answers[question_id] = st.radio(
                    "Answer",
                    ["True", "False"],
                    key=f"final-{question_id}",
                    label_visibility="collapsed",
                )
            else:
                answers[question_id] = st.text_area(
                    "Answer",
                    key=f"final-{question_id}",
                    label_visibility="collapsed",
                )
            render_source_refs(question.get("source_references", []), compact=True)
        submitted = st.form_submit_button("Submit Final Exam", type="primary")

    if submitted:
        grade = study_service().grade_final_exam(exam, answers)
        st.session_state["final_exam_grade"] = grade
        st.session_state["progress"] = progress_service().record_final_exam(current_progress(), grade)

    grade = st.session_state.get("final_exam_grade")
    if grade:
        render_grade(grade)
        with st.expander("Open Question Feedback"):
            for item in grade.get("open_feedback", []):
                st.write(f"{item['id']}. {item['feedback']}")

    st.download_button(
        "Download Exam JSON",
        json.dumps(exam, indent=2),
        file_name=f"{index.pdf_name}-final-exam.json",
        mime="application/json",
    )


def render_dashboard() -> None:
    st.subheader("Dashboard")
    index = require_index()
    sections = require_sections()
    if not index or not sections:
        return

    progress = current_progress()
    completed = set(progress.get("completed_sections", []))
    quiz_average = ProgressService.average_quiz_score(progress)
    readiness = ProgressService.exam_readiness(progress)
    readiness_status = ProgressService.readiness_status(readiness)
    readiness_action = ProgressService.readiness_action(progress)
    timing = ProgressService.timing_summary(progress, section_estimates(sections))
    next_section = next((section for section in sections if section.section_id not in completed), sections[-1])

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Overall progress", f"{progress.get('total_progress_percentage', 0):.1f}%")
    col_b.metric("Completed sections", f"{len(completed)} / {len(sections)}")
    col_c.metric("Average quiz score", f"{quiz_average:.1f}%")
    col_d.metric("Exam readiness", f"{readiness:.1f}%", readiness_status)
    st.progress(progress.get("total_progress_percentage", 0) / 100)

    st.markdown("### Recommendations")
    st.write(f"Exam readiness: **{readiness:.1f}% — {readiness_status}.** {readiness_action}")
    st.write(f"Recommended next section: **{next_section.title}** ({next_section.page_range_label()})")
    st.write(f"Last studied section: **{progress.get('last_studied_section') or 'None yet'}**")

    st.markdown("### Study Time")
    time_cols = st.columns(4)
    time_cols[0].metric("Total study time", format_duration(timing["total_seconds"]))
    time_cols[1].metric("Average per section", format_duration(timing["average_seconds"]))
    time_cols[2].metric("Estimated total", format_duration(timing["estimated_seconds"]))
    time_cols[3].metric("Actual vs estimated", format_duration(abs(timing["difference_seconds"])), "over" if timing["difference_seconds"] > 0 else "under")
    if timing["longer_than_expected"]:
        labels = [section.title for section in sections if section.section_id in timing["longer_than_expected"]]
        st.caption("Sections that took longer than expected: " + ", ".join(labels))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Weak Topics**")
        render_topic_list(progress.get("weak_topics", []))
    with col_b:
        st.markdown("**Strong Topics**")
        render_topic_list(progress.get("strong_topics", []))

    review_sections = progress.get("review_sections", [])
    if review_sections:
        st.markdown("**Recommended Review Sections**")
        names = [section.title for section in sections if section.section_id in review_sections]
        render_topic_list(names or review_sections)

    repeated = repeated_mistakes(progress)
    if repeated:
        st.markdown("**Most Repeated Mistakes**")
        for topic, count in repeated[:5]:
            st.write(f"- {topic}: {count} time(s)")

    st.markdown("### Section Progress")
    for section in sections:
        score = progress.get("quiz_scores", {}).get(section.section_id)
        understanding_score = progress.get("understanding_scores", {}).get(section.section_id)
        seconds = progress.get("section_time_seconds", {}).get(section.section_id, 0)
        cols = st.columns([3, 1, 1])
        cols[0].write(section.title)
        cols[1].write("Done" if section.section_id in completed else "Open")
        quiz_label = f"Quiz {score:.1f}%" if score is not None else "No quiz"
        cols[2].write(f"{quiz_label} | Check {understanding_score:.1f}% | {format_duration(seconds)}" if understanding_score is not None else f"{quiz_label} | {format_duration(seconds)}")

    for section in sections:
        cached_exam_focus(section, index)
        cached_flashcards(section, index)
    export_markdown = study_service().export_study_pack_markdown(
        index,
        sections,
        progress,
        {
            "exam_focus": st.session_state.get("exam_focus", {}),
            "flashcards": st.session_state.get("flashcards", {}),
            "final_exam_grade": st.session_state.get("final_exam_grade"),
        },
    )
    st.download_button(
        "Export Study Pack",
        export_markdown,
        file_name=f"{index.pdf_name}-study-pack.md",
        mime="text/markdown",
        type="primary",
    )


def render_rag_answer(result) -> None:
    st.markdown("### Answer")
    escaped = html.escape(sanitize_visible_text(result.answer, remove_files=False)).replace("\n", "<br>")
    st.markdown(f"<div class='answer-box'>{escaped}</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("Found in PDF", "Yes" if result.found else "No")
    col_b.metric("Confidence", result.confidence_label, f"{result.confidence:.4f}")
    if result.sources:
        with st.expander("Relevant Source Snippets", expanded=True):
            for source in result.sources:
                st.markdown(f"<span class='source-badge'>{html.escape(source.label())}</span>", unsafe_allow_html=True)
                st.write(sanitize_visible_text(source.text, remove_files=False))


def render_grade(grade: dict) -> None:
    st.markdown("### Results")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Score", f"{grade.get('score_percentage', 0):.1f}%")
    col_b.metric("Correct", f"{grade.get('correct_count', 0)} / {grade.get('total_questions', 0)}")
    col_c.metric("Needs review", len(grade.get("weak_topics", [])))

    with st.expander("Correct and Wrong Answers", expanded=True):
        for item in grade.get("results", []):
            status = "Correct" if item["correct"] else "Review"
            st.write(f"**{item['id']}. {status}**")
            st.write(f"Your answer: {sanitize_visible_text(item.get('student_answer') or 'No answer', remove_files=False)}")
            st.write(f"Correct answer: {sanitize_visible_text(item.get('correct_answer'), remove_files=False)}")
            st.caption(sanitize_visible_text(item.get("explanation", ""), remove_files=False))
            render_source_refs(item.get("source_references", []), compact=True)

    if grade.get("weak_topics"):
        st.markdown("**Topics that need review**")
        render_topic_list(grade["weak_topics"])


def render_concepts(concepts: list[str]) -> None:
    if not concepts:
        return
    clean_concepts = normalize_bullets(concepts, fallback="", limit=12)
    clean_concepts = [concept for concept in clean_concepts if concept]
    pills = "".join(f"<span class='pill'>{html.escape(concept)}</span>" for concept in clean_concepts)
    st.markdown(pills, unsafe_allow_html=True)


def render_topic_list(topics: list[str]) -> None:
    if not topics:
        st.caption("No topics yet.")
        return
    render_concepts([str(topic) for topic in topics[:12]])


def repeated_mistakes(progress: dict) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in progress.get("mistake_history", []):
        topics = item.get("topics") or ["General"]
        for topic in topics:
            label = str(topic)
            counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def render_source_refs(refs: list[dict], compact: bool = False) -> None:
    if not refs:
        return
    labels = [
        format_source_label(source_metadata_from_ref(ref) or ref)
        for ref in refs[:3]
    ]
    label = " ".join(
        f"<span class='source-badge'>{html.escape(source_label)}</span>"
        for source_label in labels
    )
    if compact:
        st.markdown(label, unsafe_allow_html=True)
    else:
        st.markdown(label, unsafe_allow_html=True)


def render_navigation() -> str:
    labels = study_mode_tabs()
    current = active_page()
    if st.session_state.get("nav_page") not in labels or st.session_state.get("nav_page") != current:
        st.session_state["nav_page"] = current
    try:
        selected = st.segmented_control(
            "Navigation",
            labels,
            default=current,
            label_visibility="collapsed",
            key="nav_page",
        )
    except Exception:
        selected = st.radio(
            "Navigation",
            labels,
            index=labels.index(current),
            horizontal=True,
            label_visibility="collapsed",
            key="nav_page",
        )
    selected = selected or current
    st.session_state["active_page"] = selected
    return selected


def main() -> None:
    st.set_page_config(page_title="Smart Study Assistant", layout="wide")
    apply_theme()
    init_state()
    render_header()

    page = render_navigation()
    if page == "Upload":
        render_upload()
    elif page == "Study Plan":
        render_study_plan()
    elif page == "Study Mode":
        render_study_mode()
    elif page == "Ask AI":
        render_ask()
    elif page == "Final Exam":
        render_final_exam()
    elif page == "Dashboard":
        render_dashboard()


if __name__ == "__main__":
    main()
