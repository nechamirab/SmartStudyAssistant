from __future__ import annotations

import csv
import html
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from generation.answer_generator import AnswerGenerator
from generation.mcq_generator import MCQGenerator
from ingestion.document_loader import DocumentLoader
from ingestion.ocr_loader import OCRDependencyError, OCRLoader
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingError, EmbeddingService
from services.retrieval_service import RetrievalError, RetrievalService
from vectorstores.factory import VectorStoreFactory
from vectorstores.base import VectorStoreError


RESULTS_DIR = ROOT / "experiments" / "results"

st.set_page_config(
    page_title="Smart Study Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    """Presentation polish without changing backend behavior."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        [data-testid="stSidebar"] {
            background: #f7f8fb;
            border-right: 1px solid #e3e7ee;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .hero {
            border: 1px solid #e3e7ee;
            border-radius: 8px;
            padding: 18px 20px;
            background: #ffffff;
            margin-bottom: 14px;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.05rem;
        }
        .hero p {
            margin: 0;
            color: #4b5563;
            font-size: 1rem;
        }
        .card {
            border: 1px solid #e3e7ee;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            margin-bottom: 12px;
        }
        .answer-card {
            border: 1px solid #d8e0ea;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
            line-height: 1.58;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.62rem;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid #d1d5db;
            background: #f9fafb;
            color: #374151;
        }
        .badge-high {
            background: #ecfdf5;
            border-color: #a7f3d0;
            color: #047857;
        }
        .badge-medium {
            background: #fffbeb;
            border-color: #fde68a;
            color: #92400e;
        }
        .badge-low {
            background: #fef2f2;
            border-color: #fecaca;
            color: #b91c1c;
        }
        .muted {
            color: #6b7280;
            font-size: 0.92rem;
        }
        .small-label {
            color: #4b5563;
            font-size: 0.84rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("documents_processed", 0)
    st.session_state.setdefault("debug_mode", False)
    st.session_state.setdefault("last_retrieval_results", [])
    st.session_state.setdefault("last_query", "")


def latest_file(pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            continue
    return values


def last_benchmark_accuracy() -> str:
    latest_csv = latest_file("*.csv")
    if not latest_csv:
        return "n/a"
    values = numeric_values(read_csv_rows(latest_csv), "accuracy")
    return f"{sum(values) / len(values):.3f}" if values else "n/a"


def index_summary(index: dict[str, Any] | None) -> dict[str, Any]:
    if not index:
        return {
            "document": "No document indexed",
            "pages": 0,
            "chunks": 0,
            "provider": "mock",
            "model": "mock-hash-v1",
            "dimension": "-",
            "vector_store": "memory",
        }
    embedding_service = index["embedding_service"]
    return {
        "document": index["filename"],
        "pages": len(index["pages"]),
        "chunks": len(index["chunks"]),
        "provider": embedding_service.provider,
        "model": embedding_service.model,
        "dimension": embedding_service.embedding_dimension or "unknown",
        "vector_store": index.get("vector_store_name", "memory"),
    }


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("RAG Settings")
        st.caption("Defaults are offline-safe for a live demo.")
        embedding_provider = st.selectbox(
            "Embedding provider",
            ["mock", "minilm", "e5", "bge"],
            index=0,
        )
        vector_store = st.selectbox(
            "Vector store",
            ["memory", "faiss", "chroma", "qdrant"],
            index=0,
        )
        chunk_size = st.slider("Chunk size", 100, 2000, CHUNK_SIZE, step=50)
        max_overlap = max(0, chunk_size - 1)
        overlap = st.slider(
            "Overlap",
            0,
            min(500, max_overlap),
            min(CHUNK_OVERLAP, min(500, max_overlap)),
            step=10,
        )
        top_k = st.slider("Top-K", 1, 10, 3)
        generation_mode = st.selectbox(
            "Generation mode",
            ["retrieved_chunks", "grounded_mock", "llm"],
            index=1,
        )
        show_citations = st.checkbox("Show citations", value=True)
        debug_mode = st.checkbox("Debug mode", value=st.session_state.get("debug_mode", False))
        st.session_state["debug_mode"] = debug_mode

        with st.expander("Advanced", expanded=False):
            embedding_model = st.text_input("Embedding model override", value="")
            batch_size = st.number_input("Embedding batch size", min_value=1, max_value=256, value=32)
            normalize_embeddings = st.checkbox("Normalize embeddings", value=True)

        st.divider()
        summary = index_summary(st.session_state.get("index"))
        st.write("Current index")
        st.caption(f"Document: {summary['document']}")
        st.caption(f"Embeddings: {summary['provider']} / {summary['model']}")
        st.caption(f"Vector store: {summary['vector_store']}")

    return {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model.strip(),
        "vector_store": vector_store,
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "top_k": int(top_k),
        "generation_mode": generation_mode,
        "show_citations": show_citations,
        "debug_mode": debug_mode,
        "batch_size": int(batch_size),
        "normalize_embeddings": normalize_embeddings,
    }


def render_header(settings: dict[str, Any]) -> None:
    summary = index_summary(st.session_state.get("index"))
    st.markdown(
        """
        <div class="hero">
          <h1>Smart Study Assistant — RAG Research Platform</h1>
          <p>Upload PDFs, ask grounded questions, generate quizzes, inspect retrieval, and benchmark RAG settings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_a, card_b, card_c, card_d, card_e = st.columns(5)
    card_a.metric("Documents processed", st.session_state.get("documents_processed", 0))
    card_b.metric("Total chunks", summary["chunks"])
    card_c.metric("Embedding provider", summary["provider"] or settings["embedding_provider"])
    card_d.metric("Vector store", summary["vector_store"] or settings["vector_store"])
    card_e.metric("Last accuracy", last_benchmark_accuracy())


def build_index(
    pdf_bytes: bytes,
    filename: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_bytes(pdf_bytes)
        pages = DocumentLoader().load(path)

    chunks = ChunkService(
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["overlap"],
    ).chunk_pages(pages)

    embedding_service = EmbeddingService(
        provider=settings["embedding_provider"],
        model=settings["embedding_model"],
        batch_size=settings["batch_size"],
        normalize_embeddings=settings["normalize_embeddings"],
        cache_enabled=True,
    )
    embeddings = embedding_service.embed_texts(chunks)

    store = VectorStoreFactory.create(
        settings["vector_store"],
        collection_name=filename,
    )
    store.add(chunks, embeddings)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=store,
        chunks=chunks,
    )
    return {
        "filename": filename,
        "pages": pages,
        "chunks": chunks,
        "embedding_service": embedding_service,
        "retrieval_service": retrieval_service,
        "store": store,
        "vector_store_name": settings["vector_store"],
    }


def require_index() -> dict[str, Any] | None:
    index = st.session_state.get("index")
    if not index:
        st.info("Upload and process a PDF first. For the fastest demo, use the sample PDF.")
    return index


def confidence_badge(level: str) -> None:
    level = (level or "low").lower()
    css_class = {
        "high": "badge badge-high",
        "medium": "badge badge-medium",
        "low": "badge badge-low",
    }.get(level, "badge")
    st.markdown(f"<span class='{css_class}'>Confidence: {html.escape(level)}</span>", unsafe_allow_html=True)


def render_answer_card(answer: str) -> None:
    escaped = html.escape(answer or "No answer generated.").replace("\n", "<br>")
    st.markdown(f"<div class='answer-card'>{escaped}</div>", unsafe_allow_html=True)


def render_citations(citations: list[dict[str, Any]]) -> None:
    st.subheader("Citations")
    if not citations:
        st.caption("No explicit citations were produced for this generation mode.")
        return
    for index, citation in enumerate(citations, 1):
        label = citation.get("label") or f"Citation {index}"
        with st.expander(label, expanded=index == 1):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunk", citation.get("chunk_id", "-"))
            col_b.metric("Page", citation.get("page_number", "unknown"))
            score = citation.get("score")
            col_c.metric("Score", f"{float(score):.3f}" if isinstance(score, (int, float)) else "-")
            st.caption(f"Source: {citation.get('source', 'document')}")


def retrieval_rows(results) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": item.chunk.chunk_id,
            "score": round(float(item.score), 4),
            "source": item.chunk.source_id or item.chunk.metadata.get("source", ""),
            "page": item.chunk.page_number,
        }
        for rank, item in enumerate(results, 1)
    ]


def render_retrieval_debug(results) -> None:
    if not results:
        st.info("No retrieved chunks to inspect yet.")
        return
    rows = retrieval_rows(results)
    st.dataframe(rows, width="stretch", hide_index=True)
    st.bar_chart({row["chunk_id"]: row["score"] for row in rows})
    for row, item in zip(rows, results):
        with st.expander(
            f"Rank {row['rank']} | {row['chunk_id']} | page {row['page']} | score {row['score']}"
        ):
            st.json(
                {
                    "chunk_id": item.chunk.chunk_id,
                    "source": item.chunk.source_id,
                    "page": item.chunk.page_number,
                    "score": item.score,
                    "metadata": item.chunk.metadata,
                }
            )
            st.write(item.chunk.text)


def save_quiz_outputs(questions) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "quiz.json"
    md_path = RESULTS_DIR / "quiz.md"
    MCQGenerator.save_json(questions, json_path)
    MCQGenerator.save_markdown(questions, md_path)
    return json_path, md_path


def friendly_error(message: str) -> None:
    st.error(message)
    st.caption("Tip: the default mock + memory settings do not require internet, API keys, or external services.")


def chart_by_config(rows: list[dict[str, str]], metric: str) -> None:
    values: dict[str, float] = {}
    for index, row in enumerate(rows, 1):
        try:
            value = float(row.get(metric, ""))
        except (TypeError, ValueError):
            continue
        label = (
            f"{index}. c{row.get('chunk_size', '?')}/k{row.get('top_k', '?')} "
            f"{row.get('embedding_provider', '')}"
        )
        values[label] = value
    if values:
        st.bar_chart(values)
    else:
        st.info(f"No numeric values found for {metric}.")


def best_configuration(rows: list[dict[str, str]]) -> dict[str, str] | None:
    best_row = None
    best_score = -1.0
    for row in rows:
        try:
            score = float(row.get("accuracy", ""))
        except (TypeError, ValueError):
            continue
        if score > best_score:
            best_score = score
            best_row = row
    return best_row


def render_upload_tab(settings: dict[str, Any]) -> None:
    st.subheader("Upload & Process")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    use_sample = st.button("Use sample PDF", help="Process data/example.pdf for the demo.")

    pdf_bytes = uploaded.getvalue() if uploaded else None
    filename = uploaded.name if uploaded else ""
    if use_sample:
        sample = ROOT / "data" / "example.pdf"
        pdf_bytes = sample.read_bytes()
        filename = sample.name

    if filename:
        size_kb = len(pdf_bytes or b"") / 1024
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("File", filename)
        col_b.metric("Size", f"{size_kb:.1f} KB")
        col_c.metric("Status", "Ready")
    else:
        st.info("Choose a PDF or use the sample document.")

    process_clicked = st.button("Process Document", type="primary", disabled=pdf_bytes is None)
    if use_sample or process_clicked:
        try:
            with st.spinner("Extracting text, creating chunks, embedding, and indexing..."):
                st.session_state["index"] = build_index(pdf_bytes or b"", filename, settings)
                st.session_state["documents_processed"] += 1
            summary = index_summary(st.session_state["index"])
            st.success(
                f"Processed {summary['document']}: {summary['pages']} pages, "
                f"{summary['chunks']} chunks, vector store '{summary['vector_store']}'."
            )
        except (EmbeddingError, VectorStoreError) as exc:
            friendly_error(str(exc))
        except Exception as exc:
            friendly_error(f"Document processing failed: {exc}")

    index = st.session_state.get("index")
    if index:
        summary = index_summary(index)
        st.markdown("### Processing Summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Pages", summary["pages"])
        col_b.metric("Chunks", summary["chunks"])
        col_c.metric("Vector store", summary["vector_store"])
        st.markdown("### Sample Chunks")
        for chunk in index["chunks"][:5]:
            with st.expander(f"{chunk.chunk_id} | page {chunk.page_number}"):
                st.caption(chunk.citation_label())
                st.write(chunk.text)


def render_ask_tab(settings: dict[str, Any]) -> None:
    st.subheader("Ask Questions")
    index = require_index()
    if not index:
        return

    question = st.text_area(
        "Question",
        value=st.session_state.get("last_query", ""),
        placeholder="Example: What is a sequential game?",
        height=110,
    )
    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                response = index["retrieval_service"].retrieve(question, top_k=settings["top_k"])
                st.session_state["last_query"] = question
                st.session_state["last_retrieval_results"] = response.results

                if settings["generation_mode"] == "retrieved_chunks":
                    answer = "\n\n".join(item.chunk.text for item in response.results)
                    payload = {
                        "answer": answer,
                        "citations": [],
                        "used_chunks": [item.chunk.chunk_id for item in response.results],
                        "confidence": "medium" if response.results else "low",
                        "grounded": bool(response.results),
                    }
                else:
                    contexts = AnswerGenerator.contexts_from_search_results(response.results)
                    generation = AnswerGenerator(
                        llm_provider="openai" if settings["generation_mode"] == "llm" else "mock",
                        show_citations=settings["show_citations"],
                    ).generate(question, contexts)
                    answer = generation.answer
                    payload = generation.to_dict()
                    st.session_state["last_warning"] = generation.weak_context_warning

                st.session_state["last_answer"] = answer
                st.session_state["last_generation_payload"] = payload
        except RetrievalError as exc:
            friendly_error(str(exc))
        except Exception as exc:
            friendly_error(f"Question answering failed: {exc}")

    if st.session_state.get("last_answer"):
        payload = st.session_state.get("last_generation_payload", {})
        confidence_badge(str(payload.get("confidence", "low")))
        if st.session_state.get("last_warning"):
            st.warning(st.session_state["last_warning"])
        render_answer_card(st.session_state["last_answer"])
        render_citations(payload.get("citations", []))
        if settings["debug_mode"]:
            st.subheader("Retrieved Chunks")
            render_retrieval_debug(st.session_state.get("last_retrieval_results", []))


def render_quiz_tab() -> None:
    st.subheader("Generate Quiz")
    index = require_index()
    if not index:
        return

    col_a, col_b = st.columns(2)
    count = col_a.radio("Number of questions", [5, 10], horizontal=True)
    difficulty = col_b.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    if st.button("Generate Quiz", type="primary"):
        try:
            with st.spinner("Selecting evidence and building MCQs..."):
                response = index["retrieval_service"].retrieve(
                    "main concepts definitions examples important terms",
                    top_k=10,
                )
                contexts = AnswerGenerator.contexts_from_search_results(response.results)
                questions = MCQGenerator().generate(contexts, count=count, difficulty=difficulty)
                st.session_state["quiz"] = questions
                json_path, md_path = save_quiz_outputs(questions)
            st.success(f"Quiz saved to {json_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")
        except Exception as exc:
            friendly_error(f"Quiz generation failed: {exc}")

    json_path = RESULTS_DIR / "quiz.json"
    md_path = RESULTS_DIR / "quiz.md"
    if json_path.exists() or md_path.exists():
        col_json, col_md = st.columns(2)
        if json_path.exists():
            col_json.download_button("Download JSON", json_path.read_bytes(), file_name="quiz.json")
        if md_path.exists():
            col_md.download_button("Download Markdown", md_path.read_bytes(), file_name="quiz.md")

    for index_num, question in enumerate(st.session_state.get("quiz", []), 1):
        st.markdown(f"<div class='card'><b>Question {index_num}</b><br>{html.escape(question.question)}</div>", unsafe_allow_html=True)
        for option_index, option in enumerate(question.options, 1):
            st.write(f"{option_index}. {option}")
        with st.expander("Show correct answer and citation"):
            st.success(question.correct_answer)
            st.caption(question.citation.get("label", "source unavailable"))


def render_ocr_tab() -> None:
    st.subheader("OCR")
    uploaded = st.file_uploader("Image or scanned PDF", type=["pdf", "png", "jpg", "jpeg"], key="ocr_file")
    if uploaded:
        st.caption(f"Selected: {uploaded.name} ({len(uploaded.getvalue()) / 1024:.1f} KB)")
    if st.button("Extract Text", type="primary", disabled=uploaded is None):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            try:
                with st.spinner("Running OCR extraction..."):
                    pages = OCRLoader().extract(path)
                st.success(f"Extracted text from {len(pages)} page(s).")
                st.text_area("Extracted text", "\n\n".join(page.text for page in pages), height=360)
            except OCRDependencyError as exc:
                st.error("OCR dependencies are not installed.")
                st.code("sudo apt-get install tesseract-ocr\npython -m pip install Pillow pytesseract")
                st.caption(str(exc))
            except Exception as exc:
                friendly_error(f"OCR extraction failed: {exc}")


def render_experiments_tab() -> None:
    st.subheader("Experiments")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = latest_file("*.csv")
    json_path = latest_file("*results.json") or latest_file("*.json")
    md_path = latest_file("*summary.md") or latest_file("*.md")

    if not csv_path:
        st.info("No experiment CSV found yet. Run a benchmark command below.")
    else:
        rows = read_csv_rows(csv_path)
        st.caption(f"Latest CSV: {csv_path.relative_to(ROOT)}")
        accuracy = numeric_values(rows, "accuracy")
        response_time = numeric_values(rows, "response_time")
        grounding = numeric_values(rows, "grounding_score")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rows", len(rows))
        col_b.metric("Avg accuracy", f"{sum(accuracy) / len(accuracy):.3f}" if accuracy else "n/a")
        col_c.metric("Avg grounding", f"{sum(grounding) / len(grounding):.3f}" if grounding else "n/a")
        col_d.metric(
            "Avg response",
            f"{(sum(response_time) / len(response_time)) * 1000:.1f} ms" if response_time else "n/a",
        )

        best = best_configuration(rows)
        if best:
            with st.expander("Best configuration", expanded=True):
                st.json(
                    {
                        "accuracy": best.get("accuracy"),
                        "chunk_size": best.get("chunk_size"),
                        "overlap": best.get("chunk_overlap"),
                        "top_k": best.get("top_k"),
                        "embedding_provider": best.get("embedding_provider"),
                        "embedding_model": best.get("embedding_model"),
                        "vector_store": best.get("vector_store"),
                        "generation_mode": best.get("generation_mode"),
                    }
                )

        st.markdown("### Metrics Table")
        st.dataframe(rows, width="stretch", hide_index=True)
        st.markdown("### Accuracy By Configuration")
        chart_by_config(rows, "accuracy")
        st.markdown("### Response Time By Configuration")
        chart_by_config(rows, "response_time")
        st.markdown("### Grounding Score By Configuration")
        chart_by_config(rows, "grounding_score")
        st.download_button("Download latest CSV", csv_path.read_bytes(), file_name=csv_path.name)

    if json_path:
        with st.expander(f"Latest JSON diagnostics: {json_path.name}"):
            try:
                st.json(json.loads(json_path.read_text()))
            except json.JSONDecodeError:
                st.error("Latest JSON diagnostics file is not valid JSON.")

    if md_path:
        with st.expander(f"Latest Markdown report: {md_path.name}"):
            st.markdown(md_path.read_text())

    st.markdown("### Run Command Examples")
    st.code(
        "python main_experiment.py --dataset local --chunk-size 500 --overlap 50 --top-k 3 --embedding-provider mock",
        language="bash",
    )
    st.code(
        "python main_experiment.py --dataset local --chunk-size 500 --overlap 50 --top-k 3 --generation-mode grounded_mock --show-citations",
        language="bash",
    )
    st.code(
        "python main_experiment.py --dataset ragbench --chunk-size 500 --overlap 50 --top-k 3 --embedding-provider mock",
        language="bash",
    )


def render_retrieval_tab() -> None:
    st.subheader("Retrieval Debug")
    index = require_index()
    if not index:
        return

    default_query = st.session_state.get("last_query") or "What are the main ideas?"
    query = st.text_input("Debug query", value=default_query)
    top_k = st.slider("Debug Top-K", 1, 12, 5)
    if st.button("Run Retrieval Debug", type="primary"):
        try:
            response = index["retrieval_service"].retrieve(query, top_k=top_k)
            st.session_state["last_query"] = query
            st.session_state["last_retrieval_results"] = response.results
        except Exception as exc:
            friendly_error(f"Retrieval debug failed: {exc}")

    st.caption(f"Last query: {st.session_state.get('last_query', '-')}")
    render_retrieval_debug(st.session_state.get("last_retrieval_results", []))


def render_about_tab() -> None:
    st.subheader("About")
    st.markdown(
        """
        **Retrieval-Augmented Generation (RAG)** combines document retrieval with
        answer generation. Instead of asking a model to answer from memory, the
        system first retrieves relevant chunks from uploaded study material and
        then generates or assembles an answer grounded in that evidence.

        **What this project does**

        - Processes PDFs into pages, chunks, embeddings, and vector indexes.
        - Answers questions with citations to source chunks and pages.
        - Generates study quizzes from retrieved evidence.
        - Supports optional OCR for scanned material.
        - Benchmarks RAG settings such as chunk size, Top-K, embeddings, vector
          stores, retrieval modes, and generation modes.

        **Why it is more than a PDF chatbot**

        A normal PDF chatbot mostly hides the retrieval process. This platform
        exposes retrieval scores, chunk IDs, citations, benchmark metrics, JSON
        diagnostics, and experiment reports so the system can be evaluated and
        explained academically.

        **Current limitations**

        - Mock embeddings are reliable for demos but not semantically strong.
        - Real embedding models require optional dependencies and model files.
        - OCR depends on external Tesseract installation.
        - LLM mode requires an API key; the default demo stays offline.
        - Benchmark metrics are lexical and retrieval-label dependent.

        **Next improvements**

        - Add a saved experiment comparison view.
        - Add Docker setup for one-command demos.
        - Add stronger semantic evaluation and LLM-as-judge rubrics.
        - Add richer study workflows such as flashcards and spaced repetition.
        """
    )


apply_theme()
init_state()
settings = render_sidebar()
render_header(settings)

tabs = st.tabs(
    [
        "Upload & Process",
        "Ask Questions",
        "Generate Quiz",
        "OCR",
        "Experiments",
        "Retrieval Debug",
        "About",
    ]
)

with tabs[0]:
    render_upload_tab(settings)
with tabs[1]:
    render_ask_tab(settings)
with tabs[2]:
    render_quiz_tab()
with tabs[3]:
    render_ocr_tab()
with tabs[4]:
    render_experiments_tab()
with tabs[5]:
    render_retrieval_tab()
with tabs[6]:
    render_about_tab()
