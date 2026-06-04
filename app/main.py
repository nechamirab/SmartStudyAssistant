from __future__ import annotations

from pathlib import Path

from services.exam_service import ExamGenerationError, ExamRequest, FullExamService
from services.rag_service import PDFRAGService


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "example.pdf"

    rag = PDFRAGService()
    index = rag.build_index_from_upload(pdf_path.read_bytes(), pdf_path.name)

    print("=== PDF INDEX ===")
    print(index.to_summary())

    question = "What does Game Theory study?"
    answer = rag.answer(index, question)

    print("\n=== GROUNDED ANSWER ===")
    print(f"Question: {question}")
    print(answer.answer)

    print("\n=== SOURCES ===")
    for source in answer.sources:
        print(
            f"{source.pdf_name} | page {source.page_number} | "
            f"{source.chunk_id} | score={source.score:.4f}"
        )

    print("\n=== AI QUIZ / EXAM PREVIEW ===")
    try:
        exam = FullExamService().generate_exam(
            index,
            ExamRequest(
                number_of_questions=4,
                question_types=["multiple_choice", "open_question", "true_false", "short_answer"],
            ),
        )
        print(exam)
    except ExamGenerationError as exc:
        print(exc)


if __name__ == "__main__":
    main()
