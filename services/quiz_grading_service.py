from __future__ import annotations

from typing import Any, Callable


ShortAnswerEvaluator = Callable[[dict[str, Any], Any], dict[str, Any]]


class QuizGradingService:
    """Grade section quizzes while allowing UI code to supply short-answer evaluation."""

    @staticmethod
    def grade(
        questions: list[dict[str, Any]],
        answers: dict[int, Any],
        short_answer_evaluator: ShortAnswerEvaluator | None = None,
    ) -> tuple[int, list[str]]:
        correct = 0.0
        feedback: list[str] = []

        for index, question in enumerate(questions, start=1):
            user_answer = answers.get(index)

            if question["type"] in {"multiple_choice", "true_false"}:
                if user_answer == question["answer"]:
                    correct += 1
                    feedback.append(f"Q{index}: Correct.")
                else:
                    feedback.append(f"Q{index}: Incorrect. The correct answer was: {question['answer']}")
                continue

            if short_answer_evaluator is None:
                evaluation = {"score": 0, "feedback": "Could not grade."}
            else:
                evaluation = short_answer_evaluator(question, user_answer)
            correct += float(evaluation.get("score", 0)) / 100
            feedback.append(f"Q{index}: {evaluation.get('feedback', 'Could not grade.')}")

        score = round((correct / max(1, len(questions))) * 100)
        return score, feedback
