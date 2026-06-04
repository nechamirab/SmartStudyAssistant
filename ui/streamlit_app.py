from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_RETRIEVAL_SCORE,
    RETRIEVAL_TOP_K,
)
from services.exam_service import ExamGenerationError, ExamRequest, FullExamService
from services.rag_service import PDFIndex, PDFRAGService, RAGPipelineError


st.set_page_config(page_title="Smart Study Assistant", layout="wide")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3 { letter-spacing: 0; }
        .app-note {
            border: 1px solid #d9e2ec;
            border-left: 4px solid #1f6feb;
            border-radius: 8px;
            background: #ffffff;
            padding: 14px 16px;
            margin-bottom: 16px;
            color: #243447;
        }
        .answer-box {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
            padding: 16px;
            line-height: 1.6;
        }
        .muted {
            color: #667085;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("last_answer", None)
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_exam", None)


def rag_service() -> PDFRAGService:
    return PDFRAGService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=RETRIEVAL_TOP_K,
        min_score=MIN_RETRIEVAL_SCORE,
    )


def current_index() -> PDFIndex | None:
    return st.session_state.get("index")


def require_index() -> PDFIndex | None:
    index = current_index()
    if not index:
        st.info("Upload and process a PDF first.")
    return index


def render_header() -> None:
    st.title("Smart Study Assistant")
    st.markdown(
        """
        <div class="app-note">
        This assistant answers only from uploaded PDF documents. If the answer is not found in the PDF,
        it will say: <strong>I could not find this information in the uploaded PDF.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    index = current_index()
    if not index:
        st.caption("No PDF indexed yet.")
        return

    summary = index.to_summary()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("PDF", summary["pdf_name"])
    col_b.metric("Pages", summary["pages"])
    col_c.metric("Chunks", summary["chunks"])
    col_d.metric("Top-K", RETRIEVAL_TOP_K)
    st.caption(summary["vector_store_note"])


def render_upload() -> None:
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Choose one or more PDFs", type=["pdf"], accept_multiple_files=True)
    ocr_mode = st.selectbox(
        "OCR mode",
        ["auto", "off", "force"],
        index=0,
        help="Auto uses normal PDF text first and OCRs pages with little or no extracted text.",
    )
    uploads = [(uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files]

    if uploads:
        total_bytes = sum(len(pdf_bytes) for _filename, pdf_bytes in uploads)
        col_a, col_b = st.columns(2)
        col_a.metric("Selected files", len(uploads))
        col_b.metric("Total size", f"{total_bytes / 1024:.1f} KB")
        with st.expander("Selected PDFs"):
            for filename, pdf_bytes in uploads:
                st.write(f"{filename} ({len(pdf_bytes) / 1024:.1f} KB)")

    if st.button("Process PDFs", type="primary", disabled=not uploads):
        try:
            with st.spinner("Extracting text, running OCR when needed, chunking pages, and indexing PDFs..."):
                st.session_state["index"] = rag_service().build_index_from_uploads(uploads, ocr_mode=ocr_mode)
                st.session_state["last_answer"] = None
                st.session_state["last_sources"] = []
                st.session_state["last_exam"] = None
            summary = st.session_state["index"].to_summary()
            extraction = summary["extraction"]
            st.success(
                f"Processed {summary['pdf_name']} with {summary['pages']} pages "
                f"and {summary['chunks']} chunks. OCR pages: {extraction['pages_using_ocr']}."
            )
        except RAGPipelineError as exc:
            st.error(str(exc))

    index = current_index()
    if index:
        st.markdown("### Processing Summary")
        extraction = index.extraction_summary()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Pages processed", extraction["pages_processed"])
        col_b.metric("Normal text pages", extraction["pages_using_normal_text"])
        col_c.metric("OCR pages", extraction["pages_using_ocr"])
        col_d.metric("Characters", extraction["total_characters_extracted"])
        st.json(index.to_summary())
        with st.expander("Preview first chunks"):
            for chunk in index.chunks[:5]:
                st.caption(chunk.citation_label())
                st.write(chunk.text)


def render_ask() -> None:
    st.subheader("Ask from PDF")
    index = require_index()
    if not index:
        return

    question = st.text_area(
        "Question",
        placeholder="Ask something that should be answered from the uploaded PDF.",
        height=120,
    )
    if st.button("Ask PDF", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving PDF chunks and generating a grounded answer..."):
                result = rag_service().answer(index, question)
                st.session_state["last_answer"] = result
                st.session_state["last_sources"] = result.sources
        except RAGPipelineError as exc:
            st.error(str(exc))

    result = st.session_state.get("last_answer")
    if result:
        st.markdown("### Answer")
        escaped = html.escape(result.answer).replace("\n", "<br>")
        st.markdown(f"<div class='answer-box'>{escaped}</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Found in PDF", "Yes" if result.found else "No")
        col_b.metric("Confidence", f"{result.confidence:.4f}")


def render_exam() -> None:
    st.subheader("Generate AI Quiz / Exam")
    index = require_index()
    if not index:
        return

    col_a, col_b = st.columns(2)
    number_of_questions = col_a.number_input("Number of questions", min_value=1, max_value=40, value=10)
    difficulty = col_b.selectbox("Difficulty", ["mixed", "easy", "medium", "hard"])
    question_types = st.multiselect(
        "Question types",
        ["multiple_choice", "true_false", "short_answer", "open_question"],
        default=["multiple_choice", "true_false", "short_answer", "open_question"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    include_answer_key = st.checkbox("Include answer key", value=True)

    st.caption("AI Quiz / Exam generation uses Groq and only the retrieved PDF chunks as context.")

    if st.button("Generate AI Quiz / Exam", type="primary", disabled=not question_types):
        request = ExamRequest(
            number_of_questions=int(number_of_questions),
            question_types=question_types,
            difficulty=difficulty,
            include_answer_key=include_answer_key,
        )
        try:
            with st.spinner("Groq is generating questions from retrieved PDF chunks..."):
                st.session_state["last_exam"] = FullExamService().generate_exam(index, request)
        except ExamGenerationError as exc:
            st.error(str(exc))

    exam = st.session_state.get("last_exam")
    if exam:
        if exam.get("fallback_used"):
            st.warning(exam.get("fallback_note", "Test fallback was used."))
        st.markdown(f"### {exam.get('title', 'AI Quiz / Exam')}")
        for question in exam.get("questions", []):
            st.markdown(
                f"**{question.get('id')}. {question.get('question', '')}**  \n"
                f"Type: `{question.get('type', '')}` | Difficulty: `{question.get('difficulty', '')}`"
            )
            options = question.get("options") or []
            for label, option in zip("ABCD", options):
                st.write(f"{label}. {option}")
            refs = question.get("source_references") or []
            if refs:
                st.caption(
                    "Sources: "
                    + ", ".join(
                        f"page {ref.get('page_number')}, chunk {ref.get('chunk_id')}"
                        for ref in refs
                    )
                )
            st.divider()

        if exam.get("answer_key"):
            with st.expander("Answer Key", expanded=include_answer_key):
                for item in exam["answer_key"]:
                    st.write(f"{item.get('id')}. {item.get('answer')}")

        st.download_button(
            "Download AI Quiz / Exam JSON",
            json.dumps(exam, indent=2),
            file_name=f"{index.pdf_name}-ai-quiz-exam.json",
            mime="application/json",
        )


def render_sources() -> None:
    st.subheader("View Sources")
    sources = st.session_state.get("last_sources") or []
    if not sources:
        st.info("Ask a question first to inspect retrieved chunks.")
        return

    for rank, source in enumerate(sources, 1):
        with st.expander(
            f"Source {rank}: {source.pdf_name}, page {source.page_number}, "
            f"chunk {source.chunk_id}, score {source.score:.4f}",
            expanded=rank == 1,
        ):
            st.write(source.text)


apply_theme()
init_state()
render_header()

tabs = st.tabs(["Upload PDF", "Ask from PDF", "Generate AI Quiz / Exam", "View Sources"])
with tabs[0]:
    render_upload()
with tabs[1]:
    render_ask()
with tabs[2]:
    render_exam()
with tabs[3]:
    render_sources()
