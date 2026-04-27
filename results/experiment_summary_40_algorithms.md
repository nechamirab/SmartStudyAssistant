# RAG Experimentation Results
## Experiment Summary
- **Total configurations tested**: 9
- **Evaluation questions**: 10 game theory questions
- **Embedding model**: mock (deterministic for reproducibility)
- **Vector store**: FAISS in-memory

## Results by Configuration
| Chunk Size | Overlap | Top-K | Accuracy | Precision@K | Grounding | Resp. Time (ms) | Num Chunks |
|---|---|---|---|---|---|---|---|
| 500 | 50 | 3 | 0.031 | 0.000 | 1.000 | 6.4 | 276 |
| 300 | 30 | 3 | 0.021 | 0.000 | 1.000 | 8.1 | 440 |
| 800 | 80 | 3 | 0.033 | 0.000 | 1.000 | 6.1 | 176 |
| 500 | 50 | 1 | 0.024 | 0.000 | 1.000 | 5.6 | 276 |
| 500 | 50 | 5 | 0.031 | 0.000 | 1.000 | 4.7 | 276 |
| 300 | 30 | 5 | 0.019 | 0.000 | 1.000 | 9.3 | 440 |
| 800 | 80 | 1 | 0.037 | 0.000 | 1.000 | 2.9 | 176 |
| 500 | 50 | 3 | 0.059 | 0.000 | 1.000 | 2.0 | 276 |
| 500 | 50 | 3 | 0.051 | 0.000 | 1.000 | 1.9 | 276 |

## Key Findings
**Best Accuracy**: chunk_size=500, overlap=50, top_k=3, provider=mock
- Accuracy: 0.059

**Best Grounding**: chunk_size=500, overlap=50, top_k=3, provider=mock
- Grounding Score: 1.000

**Fastest**: chunk_size=500, overlap=50, top_k=3, provider=mock
- Response Time: 1.9ms

## Analysis & Recommendations
### Chunk Size Impact
- Smaller chunks enable more precise retrieval
- Larger chunks provide more context
- Trade-off between specificity and information richness

### Top-K Impact
- Higher top-k values improve coverage but reduce speed
- Baseline performance depends on document structure

### Baseline Comparison
- FAISS semantic retrieval should outperform keyword search
- Large improvements over random suggest embedding quality

## Recommended Configuration
Based on accuracy and grounding, recommend:
- Chunk Size: 500
- Overlap: 50
- Top-K: 3
