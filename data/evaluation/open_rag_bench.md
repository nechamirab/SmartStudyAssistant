# Open RAG Bench Dataset

This project can run experiments against Vectara's Open RAG Bench dataset.

Open RAG Bench is distributed on Hugging Face as `vectara/open_ragbench`.
The dataset follows a BEIR-like layout:

```text
pdf/arxiv/
├── answers.json
├── corpus/
├── pdf_urls.json
├── qrels.json
└── queries.json
```

Download or clone the dataset into `data/open-rag-bench/`, then run:

```bash
python main_experiment.py \
  --dataset ragbench \
  --open-rag-bench-path data/open-rag-bench/pdf/arxiv \
  --limit 100
```

Use `--limit 0` to evaluate every available question. The full dataset
is large, so the default command limits the first run to 100 questions.

Results are written to:

```text
results/experiment_results_open_rag_bench.csv
results/experiment_summary_open_rag_bench.md
```
