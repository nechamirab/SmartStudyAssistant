from __future__ import annotations

from generation.base import Citation, RetrievedContext


class CitationFormatter:
    """Create stable citation labels from retrieved chunks."""

    @staticmethod
    def citation_for(context: RetrievedContext, index: int) -> Citation:
        source = context.source or str(context.metadata.get("source", "") or "document")
        page = context.page_number
        page_part = f"p. {page}" if page is not None else "page unknown"
        label = f"[{index}] {source}, {page_part}, chunk {context.chunk_id}"
        return Citation(
            chunk_id=context.chunk_id,
            label=label,
            source=source,
            page_number=page,
            score=context.score,
        )

    @staticmethod
    def inline_marker(index: int) -> str:
        return f"[{index}]"

    @staticmethod
    def render_citation_block(citations: list[Citation]) -> str:
        if not citations:
            return ""
        lines = ["Sources:"]
        for citation in citations:
            lines.append(f"- {citation.label} (score={citation.score:.3f})")
        return "\n".join(lines)
