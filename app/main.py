from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from services.chunk_service import ChunkService
from services.pdf_service import PdfService


def main():
    """
    Entry point of the application.

    Executes the pipeline:
    1. Extract PDF pages
    2. Split into chunks
    3. Print sample output
    """
    pdf_service = PdfService()
    chunk_service = ChunkService(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    pages = pdf_service.extract_pages("data/example.pdf")
    chunks = chunk_service.chunk_pages(pages)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(chunk.text[:300])


if __name__ == "__main__":
    main()