"""
Streamlit UI for Smart Study Assistant.

A clean, presentation-ready interface for document upload,
question answering, quiz generation, and text extraction.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from core.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_PROVIDER
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.pdf_service import PdfExtractionError, PdfService
from services.qa_service import QAService, QAError
from services.quiz_service import QuizService
from services.retrieval_service import RetrievalService
from services.vector_store_service import VectorStoreService


st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retrieval_service" not in st.session_state:
        st.session_state.retrieval_service = None
    if "qa_service" not in st.session_state:
        st.session_state.qa_service = None
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []


def load_pdf(uploaded_file) -> dict[str, object]:
    try:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / uploaded_file.name
            path.write_bytes(uploaded_file.getbuffer())

            pdf_service = PdfService()
            pages = pdf_service.extract_pages(str(path))
            if not pages:
                return {"success": False, "message": "No text could be extracted from this PDF."}

            chunk_service = ChunkService(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = chunk_service.chunk_pages(pages)
            if not chunks:
                return {"success": False, "message": "The PDF contains no chunkable text."}

            embedding_service = EmbeddingService()
            embeddings = embedding_service.embed_texts(chunks)
            vector_store = VectorStoreService()
            vector_store.add(chunks, embeddings)

            retrieval_service = RetrievalService(
                embedding_service=embedding_service,
                vector_store=vector_store,
            )
            qa_service = QAService(retrieval_service)

            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.vector_store = vector_store
            st.session_state.retrieval_service = retrieval_service
            st.session_state.qa_service = qa_service
            st.session_state.quiz_questions = []

            return {
                "success": True,
                "message": f"PDF loaded successfully: {uploaded_file.name}",
                "page_count": len(pages),
                "chunk_count": len(chunks),
            }

    except PdfExtractionError as error:
        return {"success": False, "message": f"PDF extraction failed: {error}"}
    except Exception as error:
        return {"success": False, "message": f"Unexpected error: {error}"}


def ask_question(query: str) -> dict[str, object]:
    if not st.session_state.pdf_loaded or not st.session_state.qa_service:
        return {"success": False, "message": "Please upload a PDF before asking questions."}

    try:
        response = st.session_state.qa_service.answer(query)
        return {
            "success": True,
            "query": response.query,
            "answer": response.answer,
            "sources": response.sources,
        }
    except QAError as error:
        return {"success": False, "message": str(error)}
    except Exception as error:
        return {"success": False, "message": f"Unexpected error: {error}"}


def generate_quiz() -> list[dict[str, object]]:
    if not st.session_state.pdf_loaded:
        return []

    questions = QuizService.generate_mcq(st.session_state.chunks, num_questions=3)
    st.session_state.quiz_questions = questions
    return [
        {
            "prompt": question.prompt,
            "options": question.options,
            "answer": question.answer,
        }
        for question in questions
    ]


initialize_state()

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 1rem;">
        <h1>📚 Smart Study Assistant</h1>
        <p style="color:#555;">Clean RAG platform for PDF question answering and quiz generation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_tab, ask_tab, quiz_tab, ocr_tab, benchmark_tab, about_tab = st.tabs(
    [
        "📄 Upload PDF",
        "❓ Ask Questions",
        "📝 Generate Quiz",
        "🔎 OCR / Text",
        "📊 Benchmark Results",
        "ℹ️ About",
    ]
)

with upload_tab:
    st.header("Upload PDF")
    st.write("Upload a document to prepare the study assistant.")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded:
        result = load_pdf(uploaded)
        if result["success"]:
            st.success(result["message"])
            st.metric("Pages", result["page_count"], delta=None)
            st.metric("Chunks", result["chunk_count"], delta=None)
        else:
            st.error(result["message"])

    st.markdown("---")
    st.subheader("Current pipeline")
    st.write(f"Embedding provider: **{EMBEDDING_PROVIDER}**")
    if st.session_state.pdf_loaded:
        st.success(f"Loaded {st.session_state.pdf_name}")
    else:
        st.info("No document loaded yet.")

with ask_tab:
    st.header("Ask Questions")
    if not st.session_state.pdf_loaded:
        st.info("Upload a PDF first to use the question answering feature.")
    else:
        query = st.text_input("Enter a question about the document")
        if st.button("Get Answer"):
            if query:
                with st.spinner("Searching the document..."):
                    answer_data = ask_question(query)
                if answer_data["success"]:
                    st.success("Answer ready")
                    st.markdown("### Answer")
                    st.write(answer_data["answer"])
                    st.markdown("### Sources")
                    for idx, source in enumerate(answer_data["sources"], start=1):
                        with st.expander(f"Source chunk {idx}"):
                            st.write(source)
                else:
                    st.error(answer_data["message"])

with quiz_tab:
    st.header("Generate Quiz")
    if not st.session_state.pdf_loaded:
        st.info("Upload a PDF before generating a quiz.")
    else:
        if st.button("Create quiz"):
            st.session_state.quiz_questions = generate_quiz()

        if st.session_state.quiz_questions:
            for idx, item in enumerate(st.session_state.quiz_questions, start=1):
                st.markdown(f"### Question {idx}")
                st.write(item["prompt"])
                for option in item["options"]:
                    st.write(f"- {option}")
                st.write(f"**Answer:** {item['answer']}")
        else:
            st.info("Create a quiz after uploading a document.")

with ocr_tab:
    st.header("OCR / Text Extraction")
    if not st.session_state.pdf_loaded:
        st.info("Upload a PDF to view extracted text.")
    else:
        st.write("This tab shows the extracted text from each page.")
        for page in st.session_state.pages:
            with st.expander(f"Page {page.page_number}", expanded=False):
                st.write(page.text or "(No text extracted from this page)")

with benchmark_tab:
    st.header("Benchmark Results")
    st.write(
        "Run the local benchmark from the command line to generate a stable performance report."
    )
    st.code("python main_experiment.py --dataset local")
    st.write("Results are saved to the `results/` folder.")
    st.write(
        "The benchmark evaluates retrieval quality, grounding, and response performance."
    )

with about_tab:
    st.header("About")
    st.markdown(
        """
        Smart Study Assistant is a clean RAG platform for analyzing PDF documents.

        - Upload text-based PDFs
        - Ask grounded questions
        - Generate basic multiple-choice quizzes
        - Inspect extracted text

        This project is designed to be easy to run and present.
        """
    )
    st.markdown("### Notes")
    st.write(
        "The app currently uses local mock embeddings by default. "
        "Sentence-transformers support is available if installed."
    )
    st.write(
        "For best results, use PDFs with extractable text rather than scanned images."
    )

st.markdown("---")
st.write("Smart Study Assistant — clean and stable RAG demo.")
