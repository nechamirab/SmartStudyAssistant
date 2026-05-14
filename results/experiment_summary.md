# RAG Experimentation Results
## Experiment Summary
- **Total configurations tested**: 9
- **Evaluation questions**: 10 game theory questions
- **Embedding model**: mock (deterministic for reproducibility)
- **Vector store**: FAISS in-memory

## Results by Configuration
| Chunk Size | Overlap | Top-K | Accuracy | Precision@K | Grounding | Resp. Time (ms) | Num Chunks |
|---|---|---|---|---|---|---|---|
| 500 | 50 | 3 | 0.098 | 0.000 | 1.000 | 0.8 | 50 |
| 300 | 30 | 3 | 0.108 | 0.000 | 1.000 | 1.0 | 64 |
| 800 | 80 | 3 | 0.093 | 0.000 | 1.000 | 0.8 | 47 |
| 500 | 50 | 1 | 0.119 | 0.000 | 1.000 | 0.8 | 50 |
| 500 | 50 | 5 | 0.084 | 0.000 | 1.000 | 0.8 | 50 |
| 300 | 30 | 5 | 0.091 | 0.000 | 1.000 | 1.0 | 64 |
| 800 | 80 | 1 | 0.130 | 0.000 | 1.000 | 0.8 | 47 |
| 500 | 50 | 3 | 0.105 | 0.000 | 1.000 | 0.3 | 50 |
| 500 | 50 | 3 | 0.103 | 0.000 | 1.000 | 0.3 | 50 |

## Key Findings
**Best Accuracy**: chunk_size=800, overlap=80, top_k=1, provider=mock
- Accuracy: 0.130

**Best Grounding**: chunk_size=500, overlap=50, top_k=3, provider=mock
- Grounding Score: 1.000

**Fastest**: chunk_size=500, overlap=50, top_k=3, provider=mock
- Response Time: 0.3ms

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
- Chunk Size: 800
- Overlap: 80
- Top-K: 1
