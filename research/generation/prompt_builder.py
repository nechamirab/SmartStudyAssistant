from __future__ import annotations

from generation.base import GenerationContext
from generation.citation_formatter import CitationFormatter


class PromptBuilder:
    """Build prompts that constrain the model to retrieved evidence."""

    def build(self, context: GenerationContext) -> str:
        evidence_blocks = []
        for index, item in enumerate(context.contexts, 1):
            citation = CitationFormatter.citation_for(item, index)
            evidence_blocks.append(
                "\n".join(
                    [
                        f"Source {index}: {citation.label}",
                        f"Text: {item.text}",
                    ]
                )
            )

        evidence = "\n\n".join(evidence_blocks) or "No retrieved context."
        return (
            "You are a grounded study assistant. Answer only from the retrieved "
            "sources. If the sources are weak or insufficient, say so. Include "
            "citation markers like [1] beside claims.\n\n"
            f"Question: {context.question}\n\n"
            f"Retrieved sources:\n{evidence}\n\n"
            "Grounded answer:"
        )
