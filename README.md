# Smart Study Assistant — RAG Platform

A clean, stable Retrieval-Augmented Generation (RAG) application for PDF study support.

## What this app does
- Upload a PDF document
- Extract text and chunk the content
- Build a lightweight semantic vector store
- Answer questions using retrieved document context
- Generate simple multiple-choice quiz prompts
- Display extracted text for review
- Run a local benchmark to verify retrieval quality

## Main features
- PDF upload and text extraction
- Mock and sentence-transformers embedding support
- Semantic retrieval with cosine similarity
- Grounded answers with source citations
- Basic quiz generation from document content
- Streamlit UI for easy local demos
- Local benchmark runner for quick validation

## Installation
1. Clone the repository
```bash
git clone <repo-url>
cd SmartStudyAssistant
```
2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

## Running the app
```bash
python -m streamlit run ui/streamlit_app.py
```
Open the browser page shown in the terminal.

## Running the benchmark
```bash
python main_experiment.py --dataset local
```
The benchmark saves output to `results/benchmark_results.csv` and `results/benchmark_summary.md`.

## Example demo flow
1. Open the Streamlit app
2. Upload `data/example.pdf`
3. Ask a question in the `Ask Questions` tab
4. Generate a quiz in the `Generate Quiz` tab
5. Review extracted text in the `OCR / Text` tab
6. Run the benchmark from the command line

## Project structure
- `core/` — shared configuration and data models
- `services/` — core PDF, chunking, embedding, retrieval, QA, evaluation, quiz logic
- `ui/` — Streamlit application
- `data/` — demo PDF and evaluation dataset
- `results/` — benchmark outputs
- `tests/` — unit tests
- `main_experiment.py` — clean benchmark runner
- `requirements.txt` — project dependencies

## Known limitations
- Works best with extractable text PDFs; scanned image OCR is not enabled by default
- Quiz generation uses simple heuristics, not a full question-generation model
- Real sentence-transformers embeddings are optional and may require model download time
- Answer generation is based on retrieved chunks rather than a generative LLM

## Notes
- Keep the app simple and stable
- Do not add experimental utilities or unfinished UI controls
- Use the mock embedding provider for offline demos
