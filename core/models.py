from dataclasses import dataclass, field
from typing import Any

from services.source_utils import format_source_label

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
        return format_source_label(
            {
                **self.metadata,
                "pdf_name": self.source_id or self.metadata.get("source"),
                "page_number": self.page_number,
                "chunk_id": self.chunk_id,
            }
        )


@dataclass(frozen=True)
class StudySection:
    """A logical study section derived from uploaded PDF content."""
    section_id: str
    title: str
    page_start: int
    page_end: int
    summary: str
    key_concepts: list[str]
    estimated_minutes: int
    difficulty: str
    chunk_ids: list[str]
    content_preview: str
    source_id: str = ""

    def page_range_label(self) -> str:
        if self.page_start == self.page_end:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"
