# Smart Study Assistant Benchmark Results
A clean local benchmark report for the final RAG platform.
## Configuration
- PDF: data/example.pdf
- Evaluation dataset: data/evaluation/eval_dataset.json
- Retrieval: semantic vector search
- Answer mode: retrieved chunks
## Results
| Chunk Size | Overlap | Top K | Provider | Accuracy | Precision@K | Grounding | Avg Latency (ms) | Chunks |
|---|---|---|---|---|---|---|---|---|
| 500 | 50 | 3 | mock | 0.090 | 0.000 | 1.000 | 0.6 | 50 |

## Best Configuration
- Best accuracy: chunk_size=500, overlap=50, top_k=3, provider=mock (0.090)
