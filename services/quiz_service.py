from dataclasses import dataclass
from typing import List

from services.llm_service import LLMService
from services.retrieval_service import RetrievalService


class QuizError(Exception):
    """Raised when quiz generation fails."""
    pass


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    answer: str


class QuizService:
    """
    Generates quiz questions from retrieved document chunks.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService | None = None,
    ):
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service or LLMService()

    def generate_quiz(
        self,
        topic: str,
        num_questions: int = 3,
    ) -> List[QuizQuestion]:
        """
        Generate quiz questions about a topic.
        """

        retrieval_response = self.retrieval_service.retrieve(
            topic,
            top_k=3,
        )

        context = "\n\n".join(
            [r.chunk.text for r in retrieval_response.results]
        )

        prompt = self._build_prompt(
            topic,
            context,
            num_questions,
        )

        response = self.llm_service.generate(prompt)

        return self._parse_questions(response)

    @staticmethod
    def _build_prompt(
        topic: str,
        context: str,
        num_questions: int,
    ) -> str:
        return (
            "You are a study assistant.\n"
            "Generate quiz questions based ONLY on the context below.\n"
            "Do not use outside knowledge.\n\n"
            f"Generate {num_questions} short questions and answers.\n\n"
            "Format:\n"
            "Q: question\n"
            "A: answer\n\n"
            f"Topic: {topic}\n\n"
            f"Context:\n{context}\n\n"
        )

    @staticmethod
    def _parse_questions(text: str) -> List[QuizQuestion]:
        """
        Parse LLM response into structured quiz questions.
        """

        lines = text.splitlines()

        questions = []

        current_question = None

        for line in lines:
            line = line.strip()

            if line.startswith("Q:"):
                current_question = line.replace("Q:", "").strip()

            elif line.startswith("A:") and current_question:
                answer = line.replace("A:", "").strip()

                questions.append(
                    QuizQuestion(
                        question=current_question,
                        answer=answer,
                    )
                )

                current_question = None

        return questions