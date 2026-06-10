from __future__ import annotations

import os
from dataclasses import dataclass

from core.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)

import logging
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM generation fails."""
    pass


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the LLM provider.
    """
    provider: str
    model: str
    temperature: float
    max_tokens: int


class LLMService:
    """
    Service responsible for generating text answers using an LLM.

    Supports:
    - mock provider for local development
    - OpenAI provider for real answer generation
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the configured LLM provider.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise LLMError("Prompt cannot be empty.")

        if self.config.provider == "mock":
            return self._generate_mock(prompt)

        if self.config.provider == "openai":
            try:
                return self._generate_openai(prompt)
            except LLMError as e:
                logger.warning("OpenAI failed, falling back to mock LLM: %s", e)
                return self._generate_mock(prompt)

        raise LLMError(f"Unsupported LLM provider: {self.config.provider}")

    @staticmethod
    def _generate_mock(prompt: str) -> str:
        """
        Generate a deterministic mock answer for development without API usage.
        """
        return (
            "MOCK_ANSWER: This is a placeholder answer generated without using an external LLM. "
            "It will be replaced by an OpenAI-based answer when LLM_PROVIDER is set to 'openai'."
        )

    def _generate_openai(self, prompt: str) -> str:
        """
        Generate a real answer using OpenAI.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError(
                "OpenAI package is not installed. Run: python -m pip install openai"
            ) from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is missing.")

        try:
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            message = str(e)

            if "insufficient_quota" in message or "429" in message:
                raise LLMError(
                    "OpenAI quota exceeded. Switch to LLM_PROVIDER='mock' or check billing."
                ) from e

            if "401" in message or "Unauthorized" in message:
                raise LLMError("OpenAI authentication failed. Check OPENAI_API_KEY.") from e

            raise LLMError(f"OpenAI generation failed: {e}") from e