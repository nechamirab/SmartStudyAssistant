from dataclasses import dataclass

@dataclass(frozen=True)
class DocumentPage:
    """Represents a single page extracted from a PDF."""
    page_number: int
    text: str

@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk of text derived from a document page."""
    chunk_id: str
    page_number: int
    text: str