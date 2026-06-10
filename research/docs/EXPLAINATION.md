# Smart Study Assistant - Project Documentation

## Overview

Smart Study Assistant is a Python project for studying PDF documents with a Retrieval-Augmented Generation (RAG) workflow.

The repository currently contains two layers:

- An active LangChain-based RAG pipeline used by the Streamlit UI and the newer benchmark path
- An older custom service-based pipeline that is still present for compatibility, tests, and legacy benchmarking

At a high level, the project supports:

- Uploading PDF files
- Extracting text
- Splitting text into chunks
- Creating embeddings
- Storing vectors for semantic retrieval
- Asking grounded questions about the PDF
- Generating simple quiz questions
- Running a local benchmark
- Optional OCR in the UI for scanned PDFs or images

## Current Active Architecture

The main demo path is:

`PDF -> LangChain loader -> text cleaning -> text splitting -> embeddings -> FAISS -> retrieval -> grounded answer with citations`

This active flow is implemented primarily in:

- `rag/langchain_pipeline.py`
- `ui/streamlit_app.py`
- `main_experiment.py` when `--rag-backend langchain` is used

## Legacy Architecture

The repository still includes a custom non-LangChain implementation based on:

- `services/pdf_service.py`
- `services/chunk_service.py`
- `services/embedding_service.py`
- `services/vector_store_service.py`
- `services/retrieval_service.py`
- `services/qa_service.py`

This code is still useful for:

- unit tests
- comparison
- the legacy benchmark path
- understanding the project evolution

It is not the main UI path anymore.

## Directory Structure

- `core/`: shared configuration and data model classes
- `rag/`: active LangChain-based pipeline
- `services/`: legacy and support services
- `ui/`: Streamlit app
- `tests/`: unit tests
- `data/`: sample input data and evaluation dataset
- root files: benchmark runner, dependencies, overview docs

---

## File-by-File Documentation

### Root Files

#### `README.md`
Purpose:
- Short project overview and setup guide

What it contains:
- project summary
- installation instructions
- Streamlit run instructions
- benchmark usage
- troubleshooting notes

Why it matters:
- It is the entry point for anyone trying to run the project locally.

#### `requirements.txt`
Purpose:
- Lists Python dependencies needed to run the project

What it contains:
- LangChain packages
- FAISS support
- PDF libraries
- sentence-transformers
- Streamlit
- pandas, numpy
- testing and formatting tools
- OCR dependency `pytesseract`

Why it matters:
- This file defines the runtime environment for the app and benchmark.

#### `main_experiment.py`
Purpose:
- Command-line benchmark runner

What it does:
- parses CLI arguments
- chooses benchmark backend: `legacy` or `langchain`
- runs one or more experiment configurations
- saves results to CSV and Markdown

Important behavior:
- `--rag-backend legacy` uses the older service stack
- `--rag-backend langchain` uses `LangChainRAGPipeline`

Why it matters:
- This is the main offline evaluation entry point for comparing retrieval behavior and answer quality.

#### `PROJECT_DOCUMENTATION.md`
Purpose:
- Detailed internal documentation of the current repository

Why it matters:
- Useful for project maintenance, onboarding, and future rewriting of formal academic documentation.

---

### `core/`

#### `core/__init__.py`
Purpose:
- Marks `core` as a Python package

What it does:
- No runtime logic

Why it matters:
- Allows imports such as `from core.models import DocumentChunk`.

#### `core/config.py`
Purpose:
- Stores basic shared configuration defaults for the older service stack

What it contains:
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `MOCK_EMBEDDING_DIM`

Why it matters:
- Used mainly by the legacy service layer.
- Some values overlap conceptually with the newer LangChain flow, but the Streamlit UI now manages its own active settings through session state.

#### `core/models.py`
Purpose:
- Defines lightweight project data models

Classes:
- `DocumentPage`: one extracted PDF page
- `DocumentChunk`: one text chunk derived from a page

Why it matters:
- These classes are the basic data contract for the legacy services and some tests.

---

### `rag/`

#### `rag/__init__.py`
Purpose:
- Public export module for the LangChain RAG package

What it exposes:
- `LangChainDependencyError`
- `LangChainPipelineError`
- `LangChainRAGPipeline`

Why it matters:
- Makes the pipeline easy to import from UI and benchmark code.

#### `rag/prompts.py`
Purpose:
- Stores the grounded answer prompt text for the LangChain layer

What it contains:
- `GROUNDED_ANSWER_PROMPT`

Why it matters:
- The current answer generation is deterministic and local, but this file prepares the project for future LLM integration while keeping prompt wording centralized.

#### `rag/langchain_pipeline.py`
Purpose:
- Core active RAG pipeline used by the UI

What it does:
- loads PDFs with LangChain
- cleans text
- splits documents into chunks
- creates embeddings
- builds a FAISS vector store
- retrieves relevant chunks
- generates grounded answers with citations
- rejects unreliable answers when topic coverage is weak

Key parts:

- `LangChainRAGPipeline.__init__`
  Stores the selected embedding model, chunk settings, and top-k.

- `load_pdf`
  Uses LangChain `PyPDFLoader` to load the PDF page by page.

- `clean_text`
  Normalizes whitespace, removes XML-like tags, removes noisy lines, and prepares text for chunking.

- `split_documents`
  Uses `RecursiveCharacterTextSplitter`.

- `build_vectorstore`
  Uses LangChain FAISS and embedding objects.

- `process_pdf`
  Runs the full ingestion pipeline and returns stats used by the UI.

- `retrieve`
  Performs semantic similarity search and returns chunk text, source, page, and score.

- `answer_question`
  Builds a grounded answer and citations from retrieved chunks.

- `_MockEmbeddings`
  Local fallback embedding class used when the app runs in mock mode.

Why it matters:
- This is the most important backend file in the current version of the project.
- It is the main reason the project now uses LangChain.

---

### `services/`

#### `services/__init__.py`
Purpose:
- Marks `services` as a Python package

What it does:
- No runtime logic

Why it matters:
- Needed for module imports throughout the repo.

#### `services/pdf_service.py`
Purpose:
- Legacy PDF extraction service

What it does:
- validates PDF path
- extracts text with PyMuPDF first
- falls back to pypdf if needed
- returns `DocumentPage` objects

Why it matters:
- Used by the older service stack and legacy benchmark path.
- Useful as a simpler alternative to LangChain loading.

#### `services/chunk_service.py`
Purpose:
- Legacy manual chunking service

What it does:
- normalizes extracted page text
- splits it into overlapping character-based chunks
- tries to cut near word boundaries
- returns `DocumentChunk` objects

Why it matters:
- Used by legacy retrieval and tests.
- Replaced in the active path by LangChain text splitting.

#### `services/embedding_service.py`
Purpose:
- Legacy embedding generation service

Supported modes:
- `mock`
- `sentence-transformers`
- `openai`

What it does:
- generates embeddings for chunks
- generates embeddings for queries
- provides deterministic fake vectors in mock mode

Why it matters:
- Still used by the old benchmark and service stack.
- It shows the project’s earlier approach before LangChain embeddings were added.

#### `services/vector_store_service.py`
Purpose:
- Legacy custom in-memory vector store

What it does:
- stores chunks and vectors in Python lists
- computes cosine similarity manually
- returns top-k matches

Why it matters:
- This is not FAISS.
- It remains in the repo for legacy functionality and tests.

#### `services/retrieval_service.py`
Purpose:
- Legacy retrieval layer over the custom embedding service and vector store

What it does:
- embeds the user query
- asks the vector store for most similar chunks
- returns a `RetrievalResponse`

Why it matters:
- Used by the older service-based flow and tests.

#### `services/qa_service.py`
Purpose:
- Legacy question-answering service

What it does:
- calls retrieval
- checks whether the answer is reliable
- returns a fallback response when topic coverage is weak
- otherwise builds an answer from retrieved text

Important note:
- This is a simple deterministic answer builder.
- It is not an LLM-based generator.

Why it matters:
- Still used in legacy tests.
- Conceptually similar to the reliability logic now present in the LangChain pipeline.

#### `services/quiz_service.py`
Purpose:
- Quiz generation logic

What it does:
- splits chunk text into sentences
- picks a keyword from a sentence
- replaces it with a blank
- generates answer options
- returns `QuizQuestion` objects

Capabilities:
- works with old `DocumentChunk` objects
- also works with LangChain documents and dict-like chunk payloads

Why it matters:
- Shared by the Streamlit UI for quiz generation.
- Keeps the quiz feature local and deterministic without requiring external LLM calls.

#### `services/evaluation_service.py`
Purpose:
- Defines evaluation metrics for benchmark experiments

What it calculates:
- token-level F1 accuracy
- Precision@K
- grounding score
- response time

Main classes:
- `MetricResult`
- `EvaluationResult`
- `EvaluationService`

Why it matters:
- This file is the scoring engine for the benchmark framework.

#### `services/baseline_retriever.py`
Purpose:
- Provides simple retrieval baselines for comparison

Implemented baselines:
- keyword overlap
- important-word presence
- random retrieval
- document-order retrieval

Why it matters:
- Useful in evaluation to compare semantic retrieval against simpler approaches.

#### `services/experiment_runner.py`
Purpose:
- Legacy experiment orchestration framework

What it does:
- loads the PDF and evaluation dataset
- runs chunking, embedding, retrieval, and evaluation
- aggregates metrics
- supports baseline retrieval comparison

Important note:
- This file uses the older custom service stack, not the new LangChain UI path.

Why it matters:
- Still valuable for legacy benchmarking and evaluation experiments.

---

### `ui/`

#### `ui/streamlit_app.py`
Purpose:
- Main Streamlit user interface

What it does:
- sets page layout and custom CSS
- manages `st.session_state`
- lets the user upload and process PDFs
- lets the user ask grounded questions
- displays answer cards, source cards, and debug details
- lets the user generate quizzes
- includes OCR support for scanned PDFs or images
- shows benchmark results

Key UI sections:
- hero header
- sidebar settings
- `Upload` tab
- `Ask` tab
- `Quiz` tab
- `OCR` tab
- `Results` tab
- `About` tab

Important backend connection:
- This file uses `LangChainRAGPipeline` as the main active backend.

Why it matters:
- This is the main demo application file.

---

### `tests/`

#### `tests/__init__.py`
Purpose:
- Marks `tests` as a package

Why it matters:
- Supports Python test discovery and imports.

#### `tests/test_core.py`
Purpose:
- Unit tests for basic legacy core services

What it tests:
- chunk splitting
- mock embeddings
- legacy vector store search
- token F1 accuracy scoring

Why it matters:
- Verifies the old service stack still behaves correctly.

#### `tests/test_langchain_pipeline.py`
Purpose:
- Unit tests for the new LangChain pipeline

What it tests:
- pipeline initialization
- text cleaning
- document splitting
- no-answer behavior for missing topics
- citation formatting
- friendly dependency failures

Why it matters:
- This is the main automated test coverage for the active RAG path.

#### `tests/test_qa_service.py`
Purpose:
- Unit tests for the legacy QA service

What it tests:
- reliable vs unreliable answer behavior
- fallback no-answer wording

Why it matters:
- Protects older QA logic still present in the repo.

---

### `data/`

#### `data/example.pdf`
Purpose:
- Sample PDF used for local demos and benchmark runs

Why it matters:
- Provides a fixed reference document so the app and benchmark can be tested consistently.

#### `data/evaluation/eval_dataset.json`
Purpose:
- Evaluation dataset for benchmark experiments

What it contains:
- benchmark questions
- expected answers
- expected source text

Why it matters:
- This is the benchmark ground truth used to score the RAG system.

---

## How the Main Demo Works

### 1. PDF Processing
The user uploads a PDF in the Streamlit app.

`ui/streamlit_app.py`:
- writes the file to a temporary path
- builds `LangChainRAGPipeline`
- calls `process_pdf`

`rag/langchain_pipeline.py` then:
- loads PDF pages
- cleans text
- splits documents
- creates embeddings
- builds an in-memory FAISS store

### 2. Question Answering
When the user asks a question:

- the UI calls `pipeline.answer_question`
- the pipeline retrieves relevant chunks
- it checks whether the retrieved content is related enough to the question
- if yes, it composes a short grounded answer
- if no, it returns a reliable no-answer message

### 3. Quiz Generation
The UI passes the processed chunks to `QuizService.generate_from_documents`.

The quiz generator:
- selects suitable sentences
- chooses keywords
- creates simple fill-in-the-blank style MCQs

### 4. OCR
The OCR tab:
- accepts PDFs or images
- uses `pytesseract`
- for PDF input, renders pages as images with PyMuPDF first

### 5. Benchmarking
The benchmark can run in two modes:

- `legacy`: old custom services
- `langchain`: newer pipeline closer to the UI behavior

Results are saved to:
- `results/benchmark_results.csv`
- `results/benchmark_summary.md`

---

## Active vs Legacy Files

### Active, Most Important Files
- `ui/streamlit_app.py`
- `rag/langchain_pipeline.py`
- `rag/prompts.py`
- `main_experiment.py`
- `tests/test_langchain_pipeline.py`

### Legacy but Still Relevant
- `services/pdf_service.py`
- `services/chunk_service.py`
- `services/embedding_service.py`
- `services/vector_store_service.py`
- `services/retrieval_service.py`
- `services/qa_service.py`
- `services/experiment_runner.py`

---

## Current Limitations

- The repository contains both new and old architectures, which adds some duplication.
- The main UI is LangChain-based, but not every supporting file has been migrated.
- FAISS is currently in-memory only in the active path.
- The answer generation is deterministic and retrieval-grounded, not a full external LLM chat generation flow.
- Quiz generation is heuristic and intentionally simple.
- OCR exists in the UI, but quality depends heavily on scan quality and Tesseract availability.

---

## Recommended Mental Model

If you want to understand the project quickly, read in this order:

1. `README.md`
2. `ui/streamlit_app.py`
3. `rag/langchain_pipeline.py`
4. `services/quiz_service.py`
5. `main_experiment.py`
6. `services/evaluation_service.py`
7. tests in `tests/`

If you want to understand the older architecture too, continue with:

8. `services/pdf_service.py`
9. `services/chunk_service.py`
10. `services/embedding_service.py`
11. `services/vector_store_service.py`
12. `services/retrieval_service.py`
13. `services/qa_service.py`
14. `services/experiment_runner.py`
