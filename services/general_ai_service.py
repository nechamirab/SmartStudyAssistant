from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from core.config import (
    GENERAL_AI_TEMPERATURE,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    OPENAI_MODEL,
    read_groq_api_key,
    read_openai_api_key,
)


OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
NO_GENERAL_AI_KEY_MESSAGE = (
    "Add OPENAI_API_KEY or GROQ_API_KEY to use the general AI tutor. "
    "Study Mode still works with uploaded PDFs."
)


def answer_general_question(question: str, chat_history: list | None = None) -> str:
    """Answer an open-ended study question with OpenAI first, then Groq.

    This tutor is intentionally not restricted to uploaded PDF content. PDF-
    grounded study features continue to use the existing study/RAG services.
    """
    clean_question = (question or "").strip()
    if not clean_question:
        return "Ask a question and I will help explain it."

    messages = _build_messages(clean_question, chat_history)
    openai_key = read_openai_api_key()
    if openai_key:
        return _chat_completion(
            OPENAI_CHAT_ENDPOINT,
            openai_key,
            os.getenv("OPENAI_MODEL", OPENAI_MODEL),
            messages,
            "OpenAI",
        )

    groq_key = read_groq_api_key()
    if groq_key:
        return _chat_completion(
            GROQ_CHAT_ENDPOINT,
            groq_key,
            os.getenv("GROQ_MODEL", GROQ_MODEL),
            messages,
            "Groq",
        )

    return NO_GENERAL_AI_KEY_MESSAGE


def _build_messages(question: str, chat_history: list | None) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI tutor for students. Explain clearly, use examples, "
                "and help the student reason through the topic. You may answer general "
                "questions. If the student asks for answers grounded in their uploaded PDF, "
                "suggest using Study Mode for PDF-specific help."
            ),
        }
    ]
    for item in (chat_history or [])[-10:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": question})
    return messages


def _chat_completion(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    provider_name: str,
) -> str:
    payload = {
        "model": model,
        "temperature": GENERAL_AI_TEMPERATURE,
        "max_tokens": min(LLM_MAX_TOKENS, 1200),
        "messages": messages,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SmartStudyAssistant/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return f"{provider_name} could not answer right now. {body or exc}"
    except urllib.error.URLError as exc:
        return f"{provider_name} could not be reached right now. {exc}"
    except Exception as exc:
        return f"{provider_name} tutor failed gracefully: {exc}"

    answer = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    return answer or f"{provider_name} returned an empty response. Try again with a shorter question."
