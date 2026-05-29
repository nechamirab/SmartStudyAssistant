from __future__ import annotations

import os

from generation.base import LLMClient


class OpenAILLM:
    """Optional OpenAI chat model wrapper."""

    provider_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        if not self.available():
            return ""
        try:
            from openai import OpenAI
        except ImportError:
            return ""

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from provided context. Use citation markers. "
                        "Do not invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
