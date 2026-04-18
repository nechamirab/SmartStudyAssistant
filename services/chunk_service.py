from __future__ import annotations

from core.models import DocumentChunk, DocumentPage


class ChunkingError(Exception):
    pass


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()


class ChunkService:
    """
    Service responsible for splitting document pages into text chunks.
    Supports overlapping chunks to preserve context between segments.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        """
        Split multiple pages into chunks.
        """
        all_chunks: list[DocumentChunk] = []

        for page in pages:
            page_chunks = self._chunk_single_page(page)
            all_chunks.extend(page_chunks)

        return all_chunks

    def _chunk_single_page(self, page: DocumentPage) -> list[DocumentChunk]:
        """
        Split a single page into overlapping chunks.
        """
        text = normalize_text(page.text)
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                cut = text.rfind(" ", start, end)
                if cut > start + int(self.chunk_size * 0.6):
                    end = cut

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_index += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"page_{page.page_number}_chunk_{chunk_index}",
                        page_number=page.page_number,
                        text=chunk_text,
                    )
                )

            if end >= text_length:
                break

            start = max(0, end - self.chunk_overlap)

        return chunks