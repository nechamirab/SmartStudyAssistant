from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.pdf_service import PdfService
from services.retrieval_service import RetrievalService
from services.vector_store_service import VectorStoreService


def main():
    """
    Run the document processing and retrieval pipeline:
    1. Extract PDF pages
    2. Split pages into chunks
    3. Generate embeddings
    4. Store vectors
    5. Retrieve the most relevant chunks for a query
    """
    pdf_service = PdfService()
    chunk_service = ChunkService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()

    pages = pdf_service.extract_pages("data/example.pdf")
    chunks = chunk_service.chunk_pages(pages)
    embeddings = embedding_service.embed_texts(chunks)

    vector_store.add(chunks, embeddings)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    query = "What is a sequential game?"
    retrieval_response = retrieval_service.retrieve(query, top_k=3)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings created: {len(embeddings)}")

    print("\n=== RETRIEVAL RESULTS ===")
    print(f"Query: {retrieval_response.query}")

    for result in retrieval_response.results:
        print("\n---")
        print(f"Score: {result.score:.4f}")
        print(f"Page: {result.chunk.page_number}")
        print(result.chunk.text[:200])


if __name__ == "__main__":
    main()