from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.pdf_service import PdfService
from services.vector_store_service import VectorStoreService


def main():
    """
    Run the initial document processing pipeline:
    1. Extract pages from a PDF
    2. Split pages into chunks
    3. Generate embeddings
    4. Print sample output
    """
    pdf_service = PdfService()
    chunk_service = ChunkService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embedding_service = EmbeddingService()

    pages = pdf_service.extract_pages("data/example.pdf")
    chunks = chunk_service.chunk_pages(pages)
    embeddings = embedding_service.embed_texts(chunks)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings created: {len(embeddings)}")

    for chunk, embedding in zip(chunks[:3], embeddings[:3]):
        print("\n---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Text preview: {chunk.text[:120]}")
        print(f"Embedding dimension: {len(embedding.vector)}")

    vector_store = VectorStoreService()
    vector_store.add(chunks, embeddings)

    query = "What is a sequential game?"
    query_vector = embedding_service.embed_query(query)

    results = vector_store.search(query_vector, top_k=3)

    print("\n=== SEARCH RESULTS ===")

    for res in results:
        print("\n---")
        print(f"Score: {res.score:.4f}")
        print(f"Page: {res.chunk.page_number}")
        print(res.chunk.text[:200])


if __name__ == "__main__":
    main()