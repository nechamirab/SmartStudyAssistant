from __future__ import annotations

import html
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from services.exam_service import ExamOptions, ExamService
from services.general_ai_service import GeneralAIService
from services.pdf_service import PdfExtractionError, PdfService
from services.progress_service import ProgressService
from services.quiz_service import QuizService
from services.section_state_service import SectionStateService
from services.study_service import StudySection, StudyService
from ui.state import has_pdf, persist_current_state, reset_section_outputs, section_context, source_label


def extract_pdf(uploaded_file: Any) -> None:
    pdf_bytes = uploaded_file.getvalue()
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp_file.name)

    try:
        tmp_file.write(pdf_bytes)
        tmp_file.close()
        pages = PdfService().extract_pages(str(tmp_path))
        set_pending_pdf(pdf_bytes, uploaded_file.name, pages)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def set_pending_pdf(pdf_bytes: bytes, pdf_name: str, pages: list[Any]) -> None:
    st.session_state.pending_pdf_bytes = pdf_bytes
    st.session_state.pending_pdf_name = pdf_name
    st.session_state.pending_pages = pages
    suggested = StudyService.suggest_session_count(pages)
    st.session_state.suggested_session_count = suggested
    st.session_state.selected_session_count = suggested
    st.session_state.pending_sections = StudyService().generate_study_plan_for_sessions(pages, suggested)

    if not st.session_state.pending_sections:
        raise PdfExtractionError("No readable study sessions could be created from this PDF.")

    st.session_state.upload_message = f"Processed {pdf_name}. Ready to generate a study plan."


def generate_study_plan_from_pending() -> None:
    pages = st.session_state.pending_pages
    session_count = int(st.session_state.selected_session_count or st.session_state.suggested_session_count or 1)
    sections = StudyService().generate_study_plan_for_sessions(pages, session_count)
    if not sections:
        raise PdfExtractionError("No readable study sessions could be created from this PDF.")

    st.session_state.pdf_bytes = st.session_state.pending_pdf_bytes
    st.session_state.pdf_name = st.session_state.pending_pdf_name
    st.session_state.pages = pages
    st.session_state.sections = sections
    st.session_state.section_states = SectionStateService.ensure_states(
        {},
        [section.section_number for section in sections],
    )
    st.session_state.current_section_index = 0
    st.session_state.upload_message = f"Generated {len(sections)} study sessions."
    st.session_state.progress = ProgressService.default_state()
    st.session_state.final_exam = None
    st.session_state.final_exam_answers = {}
    st.session_state.final_exam_result = None
    reset_section_outputs(sections[0])
    persist_current_state()


def generate_explanation(section: StudySection) -> str:
    text = section_context(section)
    concepts = ", ".join(section.key_concepts[:4]) or "the main ideas"
    sentences = [item.strip() for item in text.replace("\n", " ").split(".") if len(item.split()) >= 6]
    summary = " ".join(sentence + "." for sentence in sentences[:2]) or section.summary
    definitions = section.key_concepts[:3] or ["Core idea", "Example", "Review point"]
    return (
        f"**Summary**\n\n{summary}\n\n"
        f"**Key Ideas**\n\n- {concepts}\n- Connect the examples on {section.page_label.lower()} to the section title.\n\n"
        f"**Important Definitions**\n\n"
        + "\n".join(f"- {term}: define this term from the section notes." for term in definitions)
        + "\n\n**Exam Tips**\n\n"
        "- Be ready to explain the section in your own words.\n"
        "- Practice one example without looking at the PDF.\n"
        "- Review any key concept tag you cannot define quickly."
    )


def answer_section_question(section: StudySection, question: str) -> str:
    text = section_context(section)
    response = GeneralAIService().ask(
        [{"role": "user", "content": f"Use this study section as context when helpful:\n{text[:5000]}"}],
        question,
    )

    if response["ok"]:
        return f"{response['answer']}\n\n{source_label(section)}"

    if not text.strip():
        return (
            "I could not find readable text for this section yet. Re-open the PDF, check that this section has "
            "extractable text, then ask about a specific term or example from the page."
        )

    question_words = set(re.findall(r"[A-Za-z0-9]+", question.lower()))
    sentences = [item.strip() + "." for item in text.replace("\n", " ").split(".") if len(item.split()) > 6]

    best_sentence = sentences[0] if sentences else response.get("answer", "No text found.")
    for sentence in sentences:
        sentence_words = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
        if question_words & sentence_words:
            best_sentence = sentence
            break

    concepts = ", ".join(section.key_concepts[:3]) or "the main concepts"
    return (
        f"**Offline section answer**\n\n"
        f"I found this related line in the current section: {best_sentence}\n\n"
        f"Use it to review {concepts}. Good follow-up questions are: "
        f"\"Can I explain this in my own words?\", \"What example supports it?\", and "
        f"\"How could this appear on a quiz?\"\n\n"
        f"{source_label(section)}"
    )


def answer_ai_tutor(question: str, use_pdf_context: bool = False) -> dict[str, Any]:
    context_message = ""
    if use_pdf_context and has_pdf():
        context_message = all_study_context()[:6000]

    messages = list(st.session_state.ai_tutor_history)
    if context_message:
        messages.append({"role": "user", "content": f"Optional uploaded PDF context:\n{context_message}"})

    result = GeneralAIService().ask(messages, question)
    if result["ok"]:
        return result

    available = "I can use your uploaded PDF sections." if has_pdf() else "Upload a PDF to give me study context."
    fallback = (
        "AI Tutor needs `OPENAI_API_KEY` or `GROQ_API_KEY` for a full answer. "
        f"{available} For now, try asking one focused question such as "
        "\"summarize this section\", \"quiz me on the key concepts\", or "
        "\"explain the hardest term in simple words.\""
    )
    return {"ok": False, "answer": fallback, "provider": "none"}


def build_section_quiz(section: StudySection) -> list[dict[str, Any]]:
    generated = QuizService.generate_from_documents(
        [{"text": section_context(section), "source": st.session_state.pdf_name, "page": section.start_page}],
        num_questions=3,
    )
    questions: list[dict[str, Any]] = []
    if generated:
        first = generated[0]
        options = list(first.options)
        random.shuffle(options)
        questions.append(
            {
                "type": "multiple_choice",
                "question": first.prompt,
                "options": options,
                "answer": first.answer,
                "source_page": first.page or section.start_page,
            }
        )

    concept = section.key_concepts[0] if section.key_concepts else section.title
    questions.append(
        {
            "type": "true_false",
            "question": f"True or False: {concept} is discussed in this section.",
            "options": ["True", "False"],
            "answer": "True",
            "source_page": section.start_page,
        }
    )
    questions.append(
        {
            "type": "short_answer",
            "question": f"In one sentence, explain why {concept} matters in this section.",
            "options": [],
            "answer": "A strong answer should use the section text and mention the main idea clearly.",
            "source_page": section.start_page,
        }
    )
    return questions


def all_study_context() -> str:
    lines: list[str] = []
    for section in st.session_state.sections:
        lines.append(f"{section.title}\n{section_context(section)}")
    return "\n\n".join(lines)


def generate_final_exam(question_count: int, difficulty: str) -> dict[str, Any]:
    return ExamService().generate_final_exam(
        all_study_context(),
        ExamOptions(question_count=int(question_count), difficulty=difficulty),
    )


def recommended_review_sections() -> list[str]:
    if not has_pdf():
        return []
    progress = st.session_state.progress
    return [
        section.title
        for section in st.session_state.sections
        if section.section_number not in progress.completed_sections
    ][:3]


def next_recommended_section() -> StudySection | None:
    if not has_pdf():
        return None
    for section in st.session_state.sections:
        if section.section_number not in st.session_state.progress.completed_sections:
            return section
    return None


def build_weak_topic_review() -> str:
    review_sections = recommended_review_sections()
    if not review_sections:
        return "All sections are complete. Revisit the final exam answers and retake any quiz below 80%."

    quiz_average = ProgressService.quiz_average(st.session_state.progress)
    plans = ["### Weak Topic Review Plan"]
    for title in review_sections:
        section = next((item for item in st.session_state.sections if item.title == title), None)
        if section is None:
            continue
        topic = section.key_concepts[0] if section.key_concepts else section.title
        reason = "your quiz average is low" if quiz_average and quiz_average < 80 else "this section is not completed yet"
        plans.append(
            f"- **Review {html.escape(topic)}** - {reason} in Section {section.section_number}. "
            f"Re-read {section.page_label.lower()} and retake the section quiz."
        )
    return "\n".join(plans)
