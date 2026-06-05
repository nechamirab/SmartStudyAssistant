from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AIProvider:
    name: str
    api_key: str
    model: str


class GeneralAIService:
    """General tutor chat service with OpenAI-first, Groq-second provider selection."""

    GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def select_provider(self) -> AIProvider | None:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            return AIProvider("openai", openai_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            return AIProvider("groq", groq_key, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
        return None

    def ask(self, messages: list[dict[str, str]], question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"ok": False, "answer": "Please enter a question.", "provider": "none"}

        provider = self.select_provider()
        if provider is None:
            return {
                "ok": False,
                "answer": "Set OPENAI_API_KEY or GROQ_API_KEY to use the general AI tutor.",
                "provider": "none",
            }

        conversation = [
            {
                "role": "system",
                "content": "You are a clear, supportive AI tutor. Answer general study questions without PDF citations.",
            },
            *messages[-12:],
            {"role": "user", "content": question},
        ]
        try:
            if provider.name == "openai":
                answer = self._ask_openai(provider, conversation)
            else:
                answer = self._ask_groq(provider, conversation)
        except Exception as exc:
            return {
                "ok": False,
                "answer": f"The AI tutor could not answer right now: {exc}",
                "provider": provider.name,
            }
        return {"ok": True, "answer": answer, "provider": provider.name}

    def _ask_openai(self, provider: AIProvider, messages: list[dict[str, str]]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=provider.api_key)
        response = client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()

    def _ask_groq(self, provider: AIProvider, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": 0.3,
        }
        request = urllib.request.Request(
            self.GROQ_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
