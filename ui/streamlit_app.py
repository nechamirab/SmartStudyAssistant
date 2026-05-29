"""
Professional Streamlit UI for Smart Study Assistant.
"""

from __future__ import annotations

import html
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import LangChainDependencyError, LangChainPipelineError, LangChainRAGPipeline
from services.quiz_service import QuizQuestion, QuizService


st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f6f8fc 0%, #eef3f8 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            .hero-card,
            .status-card,
            .panel-card,
            .source-card,
            .answer-card,
            .quiz-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            }
            .hero-card {
                padding: 1.4rem 1.6rem;
                margin-bottom: 1rem;
            }
            .hero-title {
                font-size: 2.1rem;
                font-weight: 700;
                color: #0f172a;
                margin: 0;
            }
            .hero-subtitle {
                font-size: 0.98rem;
                color: #475569;
                margin-top: 0.35rem;
            }
            .status-card,
            .panel-card,
            .source-card,
            .answer-card,
            .quiz-card {
                padding: 1rem 1.1rem;
            }
            .status-label,
            .panel-label,
            .source-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #64748b;
                margin-bottom: 0.35rem;
            }
            .status-value,
            .panel-value,
            .source-title {
                font-size: 1.08rem;
                font-weight: 600;
                color: #0f172a;
            }
            .status-subtle,
            .source-meta,
            .answer-meta,
            .panel-subtle {
                font-size: 0.85rem;
                color: #64748b;
                margin-top: 0.28rem;
            }
            .badge {
                display: inline-block;
                padding: 0.24rem 0.6rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                margin-right: 0.35rem;
            }
            .badge-good {
                background: #dcfce7;
                color: #166534;
            }
            .badge-warn {
                background: #fef3c7;
                color: #92400e;
            }
            .badge-neutral {
                background: #e2e8f0;
                color: #334155;
            }
            .answer-text,
            .source-preview,
            .quiz-text {
                color: #1e293b;
                line-height: 1.55;
                font-size: 0.98rem;
            }
            .section-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.6rem;
            }
            .soft-note {
                color: #475569;
                font-size: 0.92rem;
                margin-top: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "pdf_loaded": False,
        "pdf_name": "",
        "rag_pipeline": None,
        "rag_stats": {},
        "page_documents": [],
        "chunk_documents": [],
        "last_answer": None,
        "last_sources": [],
        "last_chunks": [],
        "quiz_questions": [],
        "benchmark_results": None,
        "ocr_text": "",
        "embedding_provider": "minilm",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 3,
        "show_sources": True,
        "debug_mode": False,
        "quiz_count": 5,
        "quiz_difficulty": "medium",
        "upload_status": "",
        "ocr_status": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def embedding_model_name() -> str:
    if st.session_state.embedding_provider == "mock":
        return "mock"
    return "sentence-transformers/all-MiniLM-L6-v2"


def embedding_label() -> str:
    if st.session_state.embedding_provider == "mock":
        return "Mock"
    return "MiniLM"


def vector_store_status() -> str:
    return "Ready" if st.session_state.rag_pipeline is not None else "Not built"


def build_pipeline() -> LangChainRAGPipeline:
    return LangChainRAGPipeline(
        embedding_model_name=embedding_model_name(),
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
        top_k=st.session_state.top_k,
    )


def reset_document_state() -> None:
    st.session_state.pdf_loaded = False
    st.session_state.pdf_name = ""
    st.session_state.rag_pipeline = None
    st.session_state.rag_stats = {}
    st.session_state.page_documents = []
    st.session_state.chunk_documents = []
    st.session_state.last_answer = None
    st.session_state.last_sources = []
    st.session_state.last_chunks = []
    st.session_state.quiz_questions = []


def process_uploaded_pdf(uploaded_file: Any) -> dict[str, Any]:
    if uploaded_file is None:
        return {"success": False, "message": "Please upload a PDF first."}

    try:
        pipeline = build_pipeline()
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getbuffer())
            stats = pipeline.process_pdf(str(temp_path))

        st.session_state.pdf_loaded = True
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.rag_pipeline = pipeline
        st.session_state.rag_stats = stats
        st.session_state.page_documents = pipeline.documents
        st.session_state.chunk_documents = pipeline.chunks
        st.session_state.quiz_questions = []
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.session_state.last_chunks = []
        st.session_state.ocr_text = "\n\n".join(doc.page_content for doc in pipeline.documents if doc.page_content)
        return {"success": True, "message": f"Processed {uploaded_file.name}", "stats": stats}

    except LangChainDependencyError as error:
        reset_document_state()
        return {"success": False, "message": str(error)}
    except LangChainPipelineError:
        reset_document_state()
        return {"success": False, "message": "Something went wrong while processing the PDF."}
    except Exception:
        reset_document_state()
        return {"success": False, "message": "Something went wrong while processing the PDF."}


def answer_question(query: str) -> dict[str, Any]:
    pipeline = st.session_state.rag_pipeline
    if not st.session_state.pdf_loaded or pipeline is None:
        return {"success": False, "message": "Upload and process a PDF first."}

    try:
        result = pipeline.answer_question(query)
        st.session_state.last_answer = result["answer"]
        st.session_state.last_sources = result["sources"]
        st.session_state.last_chunks = result["retrieved_chunks"]
        return {"success": True, **result}
    except (LangChainDependencyError, LangChainPipelineError):
        return {"success": False, "message": "Something went wrong while answering the question."}
    except Exception:
        return {"success": False, "message": "Something went wrong while answering the question."}


def generate_quiz(num_questions: int) -> list[QuizQuestion]:
    if not st.session_state.pdf_loaded:
        return []

    questions = QuizService.generate_from_documents(st.session_state.chunk_documents, num_questions=num_questions)
    st.session_state.quiz_questions = questions
    return questions


def load_benchmark_results() -> pd.DataFrame | None:
    candidate_files = [
        PROJECT_ROOT / "experiments" / "results" / "benchmark_results.csv",
        PROJECT_ROOT / "results" / "benchmark_results.csv",
    ]
    available = [path for path in candidate_files if path.exists()]
    if not available:
        st.session_state.benchmark_results = None
        return None

    latest = max(available, key=lambda path: path.stat().st_mtime)
    try:
        dataframe = pd.read_csv(latest)
    except Exception:
        st.session_state.benchmark_results = None
        return None

    st.session_state.benchmark_results = dataframe
    return dataframe


def current_document_label() -> str:
    return st.session_state.pdf_name if st.session_state.pdf_loaded else "No document loaded"


def page_label(document: Any) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    page = metadata.get("page")
    return f"Page {page}" if page else "Page"


def chunk_preview_text(text: str, limit: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def render_status_card(label: str, value: str, subtle: str = "") -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_subtle = html.escape(subtle)
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{safe_label}</div>
            <div class="status-value">{safe_value}</div>
            <div class="status-subtle">{safe_subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_card(label: str, value: str, subtle: str = "") -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_subtle = html.escape(subtle)
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-label">{safe_label}</div>
            <div class="panel-value">{safe_value}</div>
            <div class="panel-subtle">{safe_subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_card(answer: str, citations: list[str]) -> None:
    badge_class = "badge-good" if citations else "badge-warn"
    badge_text = "Grounded" if citations else "Low confidence"
    citations_text = " • ".join(html.escape(citation) for citation in citations[:3]) if citations else "No citations"
    safe_answer = html.escape(answer)
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-meta"><span class="badge {badge_class}">{badge_text}</span></div>
            <div class="answer-text" style="margin-top:0.7rem;">{safe_answer}</div>
            <div class="answer-meta" style="margin-top:0.8rem;">{citations_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(chunk: dict[str, Any]) -> None:
    page = f"Page {chunk['page']}" if chunk.get("page") else "Page n/a"
    score = f"{chunk['score']:.4f}" if chunk.get("score") is not None else "n/a"
    preview = html.escape(chunk_preview_text(chunk.get("text", "")))
    source = html.escape(chunk.get("source", "Uploaded PDF"))
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-label">Source</div>
            <div class="source-title">{source}</div>
            <div class="source-meta">{html.escape(page)} • Score {html.escape(score)}</div>
            <div class="source-preview" style="margin-top:0.7rem;">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quiz_card(index: int, question: QuizQuestion) -> None:
    letters = ["A", "B", "C", "D"]
    options_html = "".join(
        f"<div class='quiz-text' style='margin-top:0.4rem;'><strong>{letters[idx]}.</strong> {html.escape(option)}</div>"
        for idx, option in enumerate(question.options[:4])
    )
    safe_prompt = html.escape(question.prompt)
    st.markdown(
        f"""
        <div class="quiz-card">
            <div class="panel-label">Question {index}</div>
            <div class="source-title" style="margin-bottom:0.7rem;">{safe_prompt}</div>
            {options_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Show answer"):
        st.write(question.answer)
        if question.citation:
            st.caption(question.citation)


def quiz_as_json(questions: list[QuizQuestion]) -> str:
    payload = [
        {
            "question": question.prompt,
            "options": question.options,
            "answer": question.answer,
            "citation": question.citation,
            "difficulty": st.session_state.quiz_difficulty,
        }
        for question in questions
    ]
    return json.dumps(payload, indent=2)


def quiz_as_markdown(questions: list[QuizQuestion]) -> str:
    lines = ["# Smart Study Assistant Quiz", ""]
    for index, question in enumerate(questions, start=1):
        lines.append(f"## Question {index}")
        lines.append(question.prompt)
        lines.append("")
        for option_index, option in enumerate(question.options[:4], start=1):
            letter = chr(64 + option_index)
            lines.append(f"- {letter}. {option}")
        lines.append("")
        lines.append(f"Answer: {question.answer}")
        if question.citation:
            lines.append(f"Citation: {question.citation}")
        lines.append("")
    return "\n".join(lines)


def extract_ocr_text(uploaded_file: Any) -> str:
    import fitz
    import pytesseract
    from PIL import Image

    pytesseract.get_tesseract_version()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        extracted_pages: list[str] = []
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            mode = "RGBA" if pixmap.alpha else "RGB"
            image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
            page_text = pytesseract.image_to_string(image).strip()
            if page_text:
                extracted_pages.append(f"[Page {index}]\n{page_text}")
        return "\n\n".join(extracted_pages).strip()

    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    return pytesseract.image_to_string(image).strip()


def handle_ocr_request(uploaded_file: Any) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "Please upload a PDF or image first."

    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        st.session_state.ocr_text = ""
        return False, "OCR is not installed. Install pytesseract and Pillow."

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        st.session_state.ocr_text = ""
        return False, "Tesseract is not installed."

    try:
        text = extract_ocr_text(uploaded_file)
    except Exception:
        st.session_state.ocr_text = ""
        return False, "Something went wrong during OCR extraction."

    st.session_state.ocr_text = text
    if not text.strip():
        return False, "No text was extracted."
    return True, "OCR text extracted."


inject_css()
initialize_state()

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.embedding_provider = st.selectbox(
        "Embedding provider",
        ["minilm", "mock"],
        index=["minilm", "mock"].index(st.session_state.embedding_provider),
    )
    st.session_state.chunk_size = st.slider("Chunk size", min_value=200, max_value=1000, value=st.session_state.chunk_size, step=50)
    st.session_state.chunk_overlap = st.slider(
        "Overlap",
        min_value=0,
        max_value=min(200, st.session_state.chunk_size - 1),
        value=min(st.session_state.chunk_overlap, min(200, st.session_state.chunk_size - 1)),
        step=10,
    )
    st.session_state.top_k = st.slider("Top-K", min_value=1, max_value=8, value=st.session_state.top_k, step=1)
    st.session_state.show_sources = st.checkbox("Show sources", value=st.session_state.show_sources)
    st.session_state.debug_mode = st.checkbox("Debug mode", value=st.session_state.debug_mode)

    if st.session_state.embedding_provider == "mock":
        st.warning("Mock embeddings are for testing only. Answers may be weak.")
    else:
        st.info("Using real semantic embeddings.")

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📚 Smart Study Assistant</div>
        <div class="hero-subtitle">PDF Q&amp;A • Quiz Generation • RAG with Citations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

status_columns = st.columns(4)
with status_columns[0]:
    render_status_card("Current document", current_document_label(), "Processed PDF")
with status_columns[1]:
    render_status_card("Chunks", str(st.session_state.rag_stats.get("chunks", 0)), "Ready for retrieval")
with status_columns[2]:
    render_status_card("Embedding", embedding_label(), embedding_model_name())
with status_columns[3]:
    render_status_card("Vector store", vector_store_status(), "FAISS in memory")

upload_tab, ask_tab, quiz_tab, ocr_tab, results_tab, about_tab = st.tabs(
    ["Upload", "Ask", "Quiz", "OCR", "Results", "About"]
)

with upload_tab:
    st.subheader("Upload")
    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
    upload_action_col, upload_status_col = st.columns([1, 2])
    with upload_action_col:
        process_clicked = st.button("Process PDF", use_container_width=True, disabled=uploaded_file is None)
    with upload_status_col:
        if process_clicked:
            with st.spinner("Processing PDF..."):
                result = process_uploaded_pdf(uploaded_file)
            st.session_state.upload_status = result["message"]
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
        elif st.session_state.upload_status:
            st.caption(st.session_state.upload_status)

    if st.session_state.pdf_loaded:
        pages = st.session_state.rag_stats.get("pages", 0)
        chunks = st.session_state.rag_stats.get("chunks", 0)
        model = st.session_state.rag_stats.get("embedding_model", embedding_model_name())
        metric_columns = st.columns(3)
        with metric_columns[0]:
            render_panel_card("Pages", str(pages))
        with metric_columns[1]:
            render_panel_card("Chunks", str(chunks))
        with metric_columns[2]:
            render_panel_card("Embedding model", model)

        with st.expander("Preview extracted chunks"):
            for index, chunk in enumerate(st.session_state.chunk_documents[:3], start=1):
                metadata = dict(getattr(chunk, "metadata", {}) or {})
                page = metadata.get("page")
                render_source_card(
                    {
                        "source": Path(str(metadata.get("source", st.session_state.pdf_name))).name,
                        "page": page,
                        "score": None,
                        "text": getattr(chunk, "page_content", ""),
                    }
                )

with ask_tab:
    st.subheader("Ask")
    if not st.session_state.pdf_loaded:
        st.info("Upload and process a PDF first.")
    else:
        question = st.text_input("Ask a question about the document", placeholder="What is INNER JOIN?")
        ask_clicked = st.button("Ask", use_container_width=False)

        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching the document..."):
                    answer_data = answer_question(question)

                if not answer_data["success"]:
                    st.error(answer_data["message"])
                elif not answer_data["citations"]:
                    st.warning("I could not find a reliable answer in the uploaded document.")
                    render_answer_card(answer_data["answer"], [])
                else:
                    render_answer_card(answer_data["answer"], answer_data["citations"])

        if st.session_state.last_answer and st.session_state.show_sources and st.session_state.last_sources and st.session_state.last_chunks:
            st.markdown("<div class='section-title'>Sources</div>", unsafe_allow_html=True)
            source_columns = st.columns(2)
            for index, chunk in enumerate(st.session_state.last_chunks):
                with source_columns[index % 2]:
                    render_source_card(chunk)

        if st.session_state.debug_mode and st.session_state.last_chunks:
            st.markdown("<div class='section-title'>Debug</div>", unsafe_allow_html=True)
            for index, chunk in enumerate(st.session_state.last_chunks, start=1):
                label = f"Chunk {index} • {chunk.get('source', 'Uploaded PDF')}"
                with st.expander(label):
                    st.write(chunk.get("text", ""))

with quiz_tab:
    st.subheader("Quiz")
    if not st.session_state.pdf_loaded:
        st.info("Upload and process a PDF first.")
    else:
        quiz_controls = st.columns([1, 1, 1.2])
        with quiz_controls[0]:
            st.session_state.quiz_count = st.selectbox("Questions", [5, 10], index=[5, 10].index(st.session_state.quiz_count))
        with quiz_controls[1]:
            st.session_state.quiz_difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=["easy", "medium", "hard"].index(st.session_state.quiz_difficulty))
        with quiz_controls[2]:
            if st.button("Generate Quiz", use_container_width=True):
                with st.spinner("Generating quiz..."):
                    generate_quiz(st.session_state.quiz_count)

        if st.session_state.quiz_questions:
            download_columns = st.columns(2)
            with download_columns[0]:
                st.download_button(
                    "Download JSON",
                    data=quiz_as_json(st.session_state.quiz_questions),
                    file_name="smart_study_quiz.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with download_columns[1]:
                st.download_button(
                    "Download Markdown",
                    data=quiz_as_markdown(st.session_state.quiz_questions),
                    file_name="smart_study_quiz.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            for index, question in enumerate(st.session_state.quiz_questions, start=1):
                render_quiz_card(index, question)
        else:
            st.info("Generate a quiz to view questions.")

with ocr_tab:
    st.subheader("OCR")
    ocr_file = st.file_uploader(
        "Upload scanned PDF or image",
        type=["pdf", "png", "jpg", "jpeg"],
        key="ocr_upload",
    )
    ocr_action_col, ocr_status_col = st.columns([1, 2])
    with ocr_action_col:
        extract_clicked = st.button("Extract Text", use_container_width=True, disabled=ocr_file is None)
    with ocr_status_col:
        if extract_clicked:
            with st.spinner("Running OCR..."):
                success, message = handle_ocr_request(ocr_file)
            st.session_state.ocr_status = message
            if success:
                st.success(message)
            elif message == "Tesseract is not installed.":
                st.error("Tesseract is not installed.")
                st.caption("Install the native `tesseract` binary to enable OCR.")
            elif message == "OCR is not installed. Install pytesseract and Pillow.":
                st.error("OCR is not installed.")
                st.caption("Install `pytesseract` and `Pillow` to enable OCR.")
            else:
                st.info(message)
        elif st.session_state.ocr_status:
            st.caption(st.session_state.ocr_status)

    st.text_area(
        "Extracted text",
        value=st.session_state.ocr_text,
        height=320,
        placeholder="Extracted text will appear here.",
    )

with results_tab:
    st.subheader("Results")
    results_df = load_benchmark_results()
    if results_df is None or results_df.empty:
        st.info("No benchmark results yet.")
    else:
        best_accuracy = results_df["accuracy"].max() if "accuracy" in results_df else 0.0
        best_grounding = results_df["grounding_score"].max() if "grounding_score" in results_df else 0.0
        best_response = results_df["avg_response_time_sec"].min() if "avg_response_time_sec" in results_df else 0.0

        metric_columns = st.columns(3)
        with metric_columns[0]:
            render_panel_card("Best accuracy", f"{best_accuracy:.3f}")
        with metric_columns[1]:
            render_panel_card("Grounding", f"{best_grounding:.3f}")
        with metric_columns[2]:
            render_panel_card("Response time", f"{best_response:.3f}s")

        st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.code("python main_experiment.py --dataset local")

with about_tab:
    st.subheader("About")
    left_col, right_col, limit_col = st.columns(3)
    with left_col:
        st.markdown("**What it does**")
        st.markdown("- Upload PDFs")
        st.markdown("- Ask questions")
        st.markdown("- Generate quizzes")
        st.markdown("- Show sources")
    with right_col:
        st.markdown("**How RAG works**")
        st.markdown("- PDF → Chunks")
        st.markdown("- Embeddings")
        st.markdown("- Vector Search")
        st.markdown("- Answer")
    with limit_col:
        st.markdown("**Current limitations**")
        st.markdown("- Quality depends on PDF text")
        st.markdown("- Real embeddings are recommended")
        st.markdown("- OCR quality depends on scan quality")
