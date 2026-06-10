from __future__ import annotations


def retrieval_debug_record(question: str, results) -> dict:
    """Convert retrieval results into a JSON-serializable diagnostic record."""
    return {
        "question": question,
        "retrieved_chunks": [
            {
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source_id,
                "page": item.chunk.page_number,
                "score": item.score,
                "text": item.chunk.text,
            }
            for item in results
        ],
    }
