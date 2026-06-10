# Smart Study Assistant Demo Guide

Use this checklist for the final university presentation.

## 1. Start The UI

```bash
python -m streamlit run ui/streamlit_app.py
```

Open the local Streamlit URL.

## 2. Upload And Process A PDF

- Go to **Upload & Process**.
- Click **Use sample PDF** or upload `data/example.pdf`.
- Use sidebar defaults: `mock` embeddings, `memory` vector store, chunk size `500`, overlap `50`.
- Click **Process document** and confirm page/chunk counts appear.

## 3. Ask A Grounded Question

- Go to **Ask Questions**.
- Ask: `What is a sequential game?`
- Use sidebar settings: `grounded_mock`, `Top-K = 3`, and citations enabled.
- Show the answer card, confidence badge, and citation cards.
- Enable **Debug mode** in the sidebar to show retrieved chunks and scores.

## 4. Generate A Quiz

- Go to **Generate Quiz**.
- Generate 5 medium questions.
- Confirm the app writes:
  - `experiments/results/quiz.json`
  - `experiments/results/quiz.md`

## 5. Run A Local Benchmark

```bash
python main_experiment.py --dataset local --chunk-size 500 --overlap 50 --top-k 3 --embedding-provider mock
```

Show:

- `experiments/results/experiment_results.csv`
- `experiments/results/experiment_results.json`
- `experiments/results/experiment_summary.md`

Then open the **Experiments** tab and show the metrics table, best configuration,
charts, JSON diagnostics, and Markdown report.

## 6. Run Or Explain RAGBench

If local Open RAG Bench files are available:

```bash
python main_experiment.py --dataset ragbench --open-rag-bench-path data/open-rag-bench --chunk-size 500 --overlap 50 --top-k 3 --embedding-provider mock
```

If they are not available, run:

```bash
python main_experiment.py --dataset ragbench --chunk-size 500 --overlap 50 --top-k 3 --embedding-provider mock
```

The command should fail quickly with a clear instruction to provide local files
or opt into download with `--download-ragbench`.

## 7. Optional Real Embeddings

After installing `sentence-transformers`, try:

```bash
python main_experiment.py --dataset local --chunk-size 500 --overlap 50 --top-k 3 --embedding-provider minilm
```

The project falls back to mock embeddings if the optional provider is missing or
the model cannot be loaded.

## 8. Presentation Closing

Emphasize that this is more than a PDF chatbot:

- It supports grounded answers with citations.
- It generates quizzes from retrieved evidence.
- It compares RAG configurations with exported metrics.
- It handles optional OCR, real embeddings, and vector database backends without
  requiring paid APIs for the core demo.
