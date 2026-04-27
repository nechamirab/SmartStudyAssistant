from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.pdf_service import PdfService
from services.retrieval_service import RetrievalService
from services.vector_store_service import VectorStoreService
from services.qa_service import QAService
from pathlib import Path

def main():
    """
    Run full pipeline:
    PDF → chunks → embeddings → vector store → retrieval → QA
    """

    pdf_service = PdfService()
    chunk_service = ChunkService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "example.pdf"

    pages = pdf_service.extract_pages(pdf_path)
    chunks = chunk_service.chunk_pages(pages)
    embeddings = embedding_service.embed_texts(chunks)

    vector_store.add(chunks, embeddings)

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    qa_service = QAService(retrieval_service)

    query = "What is the lecture about?"
    qa_response = qa_service.answer(query)

    print("\n=== SOURCES ===")
    for i, source in enumerate(qa_response.sources, start=1):
        print(f"\n--- Source {i} ---")
        print(source[:500])

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings created: {len(embeddings)}")

    print("\n=== ANSWER ===")
    print(f"Query: {qa_response.query}")
    print("\nAnswer:")
    print(qa_response.answer)


if __name__ == "__main__":
    main()