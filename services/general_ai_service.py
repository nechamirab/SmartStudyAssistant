from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translations import normalize_language, translate, tutor_language_instruction


@dataclass(frozen=True)
class AIProvider:
    name: str
    api_key: str
    model: str


class GeneralAIService:
    """General tutor chat service with OpenAI-first, Groq-second provider selection."""

    GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / "ui" / ".streamlit" / "_env"

    def __init__(self, env_file: str | Path | None = DEFAULT_ENV_FILE) -> None:
        self.env_file = Path(env_file) if env_file else None

    def select_provider(self) -> AIProvider | None:
        self._load_local_env()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            return AIProvider("openai", openai_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            return AIProvider("groq", groq_key, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
        return None

    def _load_local_env(self) -> None:
        if not self.env_file or not self.env_file.exists():
            return

        for line in self.env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value

    @staticmethod
    def build_tutor_prompt(language: str = "en") -> str:
        return tutor_language_instruction(language)

    def complete(
        self,
        system_prompt: str,
        prompt: str,
        language: str = "en",
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        language = normalize_language(language)
        prompt = (prompt or "").strip()
        if not prompt:
            return {"ok": False, "answer": translate("enter_question_first", language), "provider": "none"}

        provider = self.select_provider()
        if provider is None:
            return {
                "ok": False,
                "answer": (
                    "Set OPENAI_API_KEY or GROQ_API_KEY to use AI features."
                    if language == "en"
                    else "הגדירו OPENAI_API_KEY או GROQ_API_KEY כדי להשתמש ביכולות AI."
                ),
                "provider": "none",
            }

        conversation = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt},
        ]
        try:
            if provider.name == "openai":
                answer = self._ask_openai(provider, conversation, response_format=response_format)
            else:
                answer = self._ask_groq(provider, conversation, response_format=response_format)
        except Exception as exc:
            answer = (
                f"פעולת ה-AI נכשלה כרגע: {exc}"
                if language == "he"
                else f"The AI request could not complete right now: {exc}"
            )
            return {
                "ok": False,
                "answer": answer,
                "provider": provider.name,
            }
        return {"ok": True, "answer": answer, "provider": provider.name}

    def ask(self, messages: list[dict[str, str]], question: str, language: str = "en") -> dict[str, Any]:
        language = normalize_language(language)
        question = (question or "").strip()
        if not question:
            return {"ok": False, "answer": translate("enter_question_first", language), "provider": "none"}

        provider = self.select_provider()
        if provider is None:
            return {
                "ok": False,
                "answer": (
                    "Set OPENAI_API_KEY or GROQ_API_KEY to use the general AI tutor."
                    if language == "en"
                    else "הגדירו OPENAI_API_KEY או GROQ_API_KEY כדי להשתמש במורה ה-AI."
                ),
                "provider": "none",
            }

        conversation = [
            {
                "role": "system",
                "content": (
                    "You are a clear, supportive AI tutor. Answer general study questions without PDF citations.\n"
                    f"{self.build_tutor_prompt(language)}"
                ),
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
            answer = (
                f"מורה ה-AI לא הצליח לענות כרגע: {exc}"
                if language == "he"
                else f"The AI tutor could not answer right now: {exc}"
            )
            return {
                "ok": False,
                "answer": answer,
                "provider": provider.name,
            }
        return {"ok": True, "answer": answer, "provider": provider.name}

    def _ask_openai(
        self,
        provider: AIProvider,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=provider.api_key)
        request: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "temperature": 0.3,
        }
        if response_format:
            request["response_format"] = response_format
        response = client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()

    def _ask_groq(
        self,
        provider: AIProvider,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": 0.3,
        }
        if response_format:
            payload["response_format"] = response_format
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
