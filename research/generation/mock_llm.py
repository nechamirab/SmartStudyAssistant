from __future__ import annotations

from generation.base import LLMClient


class MockLLM:
    """Deterministic fallback LLM used for tests and offline experiments."""

    provider_name = "mock"

    def generate(self, prompt: str) -> str:
        # The higher-level AnswerGenerator performs deterministic grounded
        # synthesis directly. This class exists to satisfy the LLM abstraction
        # without introducing non-deterministic behavior in tests.
        return ""
