import json
import unittest
from unittest.mock import patch

from services.general_ai_service import (
    GROQ_CHAT_ENDPOINT,
    NO_GENERAL_AI_KEY_MESSAGE,
    OPENAI_CHAT_ENDPOINT,
    answer_general_question,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": self.content}}]}
        ).encode("utf-8")


class GeneralAIServiceTests(unittest.TestCase):
    def test_uses_openai_when_openai_key_exists(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, json.loads(request.data.decode("utf-8"))))
            return _FakeResponse("OpenAI answer")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key", "GROQ_API_KEY": "groq-key"}, clear=True):
            with patch("services.general_ai_service.urllib.request.urlopen", side_effect=fake_urlopen):
                answer = answer_general_question("Explain gradients")

        self.assertEqual(answer, "OpenAI answer")
        self.assertEqual(calls[0][0], OPENAI_CHAT_ENDPOINT)
        self.assertEqual(calls[0][1]["model"], "gpt-4o-mini")

    def test_falls_back_to_groq_when_openai_key_missing(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, json.loads(request.data.decode("utf-8"))))
            return _FakeResponse("Groq answer")

        with patch.dict("os.environ", {"GROQ_API_KEY": "groq-key"}, clear=True):
            with patch("services.general_ai_service.urllib.request.urlopen", side_effect=fake_urlopen):
                answer = answer_general_question("Give me a study example")

        self.assertEqual(answer, "Groq answer")
        self.assertEqual(calls[0][0], GROQ_CHAT_ENDPOINT)
        self.assertEqual(calls[0][1]["model"], "llama-3.1-8b-instant")

    def test_returns_helpful_message_when_no_api_key_exists(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("services.general_ai_service.read_groq_api_key", return_value=""):
                answer = answer_general_question("Can you help?")

        self.assertEqual(answer, NO_GENERAL_AI_KEY_MESSAGE)


if __name__ == "__main__":
    unittest.main()
