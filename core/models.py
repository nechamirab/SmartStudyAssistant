from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class DocumentPage:
    """Represents a single page extracted from a PDF."""
    page_number: int
    text: str
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk of text derived from a document page."""
    chunk_id: str
    page_number: int
    text: str
    source_id: str = ""
    start_char: int | None = None
    end_char: int | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation_label(self) -> str:
        """Return a compact source label for citation rendering."""
        source = self.source_id or str(self.metadata.get("source", "") or "document")
        return f"{source}, page {self.page_number}, chunk {self.chunk_id}"
