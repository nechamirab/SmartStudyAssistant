from __future__ import annotations

from chunking.strategies import ChunkingStrategyFactory, normalize_text
from core.models import DocumentChunk, DocumentPage


class ChunkingError(Exception):
    pass


class ChunkService:
    """
    Service responsible for splitting document pages into text chunks.
    Supports overlapping chunks to preserve context between segments.
    """
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str = "recursive",
    ):
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_name = strategy
        self.strategy = ChunkingStrategyFactory.create(strategy)

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
        spans = self.strategy.split(text, self.chunk_size, self.chunk_overlap)

        for chunk_index, span in enumerate(spans, 1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"page_{page.page_number}_chunk_{chunk_index}",
                    page_number=page.page_number,
                    text=span.text,
                    source_id=page.source_id,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    parent_id=span.parent_id,
                    metadata={
                        **page.metadata,
                        **(span.metadata or {}),
                        "chunking_strategy": self.strategy.name,
                    },
                )
            )

        return chunks
