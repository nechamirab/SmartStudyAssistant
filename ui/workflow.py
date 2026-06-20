from __future__ import annotations

import html
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from services.context_retrieval_service import ContextRetrievalService
from services.auth_service import AuthService
from services.database_service import DatabaseService
from services.exam_service import ExamOptions, ExamService
from services.general_ai_service import GeneralAIService
from services.pdf_service import PdfExtractionError, PdfService
from services.progress_service import ProgressService
from services.quiz_service import QuizService
from services.section_state_service import SectionStateService
from services.study_service import StudySection, StudyService
from translations import current_language, t, tutor_language_instruction
from ui.state import has_pdf, page_label, persist_current_state, reset_section_outputs, section_context, source_label


NOT_ENOUGH_INFORMATION = "The uploaded PDF does not contain enough information to answer this question."
GROUNDED_SYSTEM_PROMPT = (
    "You are a grounded PDF study assistant. "
    "Answer only using the provided PDF context. "
    "Do not use outside knowledge. Do not guess."
)


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
    st.session_state.pending_sections = StudyService().generate_study_plan_for_sessions(
        pages,
        suggested,
        language=current_language(),
    )
    st.session_state.pending_plan_signature = pending_study_plan_signature(pdf_name, pages, suggested, current_language())

    if not st.session_state.pending_sections:
        raise PdfExtractionError(t("no_readable_sessions"))

    st.session_state.upload_message = t("processed_pdf_ready", pdf_name=pdf_name)


def generate_study_plan_from_pending() -> None:
    pages = st.session_state.pending_pages
    session_count = int(st.session_state.selected_session_count or st.session_state.suggested_session_count or 1)
    signature = pending_study_plan_signature(
        st.session_state.pending_pdf_name,
        pages,
        session_count,
        current_language(),
    )
    if st.session_state.pending_sections and st.session_state.pending_plan_signature == signature:
        sections = st.session_state.pending_sections
    else:
        sections = StudyService().generate_study_plan_for_sessions(
            pages,
            session_count,
            language=current_language(),
        )
        st.session_state.pending_sections = sections
        st.session_state.pending_plan_signature = signature
    if not sections:
        raise PdfExtractionError(t("no_readable_sessions"))

    st.session_state.pdf_bytes = st.session_state.pending_pdf_bytes
    st.session_state.pdf_name = st.session_state.pending_pdf_name
    st.session_state.pages = pages
    st.session_state.sections = sections
    st.session_state.section_states = SectionStateService.ensure_states(
        {},
        [section.section_number for section in sections],
    )
    st.session_state.current_section_index = 0
    st.session_state.upload_message = t("generated_sessions", count=len(sections))
    st.session_state.progress = ProgressService.default_state()
    st.session_state.final_exam = None
    st.session_state.final_exam_answers = {}
    st.session_state.final_exam_result = None
    create_sqlite_session_for_current_plan()
    reset_section_outputs(sections[0])
    persist_current_state()


def pending_study_plan_signature(pdf_name: str, pages: list[Any], session_count: int, language: str) -> str:
    page_numbers = [
        str(int(getattr(page, "page_number", 0) or 0))
        for page in pages
        if (getattr(page, "text", "") or "").strip()
    ]
    text_size = sum(len(getattr(page, "text", "") or "") for page in pages)
    return f"{pdf_name}:{language}:{int(session_count or 0)}:{text_size}:{','.join(page_numbers)}"


def create_sqlite_session_for_current_plan() -> None:
    user = AuthService().current_user()
    if not user:
        st.session_state.current_db_document_id = None
        st.session_state.current_db_session_id = None
        return
    try:
        first_title = st.session_state.sections[0].title if st.session_state.sections else st.session_state.pdf_name
        document_id, session_id = DatabaseService().create_session_from_state(
            user_id=int(user["id"]),
            filename=st.session_state.pdf_name,
            title=first_title,
            language=current_language(),
            pages=st.session_state.pages,
            sections=st.session_state.sections,
            pdf_bytes=st.session_state.pdf_bytes,
        )
        st.session_state.current_db_document_id = document_id
        st.session_state.current_db_session_id = session_id
        st.session_state.db_status_message = "Study session saved."
    except Exception as exc:
        st.session_state.current_db_document_id = None
        st.session_state.current_db_session_id = None
        st.session_state.db_status_message = f"SQLite session save failed: {exc}"


def load_saved_study_session(session_id: int) -> bool:
    user = AuthService().current_user()
    if not user:
        return False
    payload = DatabaseService().load_study_session(int(user["id"]), int(session_id))
    if not payload:
        st.session_state.db_status_message = "Could not load saved session."
        return False

    session = payload["session"]
    st.session_state.pdf_bytes = payload.get("pdf_bytes") or b""
    st.session_state.pdf_name = session.get("filename", "")
    st.session_state.pages = payload["pages"]
    st.session_state.sections = payload["sections"]
    st.session_state.section_states = payload["section_states"]
    st.session_state.progress = payload["progress"]
    st.session_state.final_exam = payload["final_exam"]
    st.session_state.final_exam_answers = payload["final_exam_answers"]
    st.session_state.final_exam_result = payload["final_exam_result"]
    st.session_state.current_section_index = 0
    st.session_state.current_db_document_id = int(session["document_id"])
    st.session_state.current_db_session_id = int(session["id"])
    st.session_state.upload_message = f"Loaded saved session for {st.session_state.pdf_name}."
    st.session_state.db_status_message = "Saved session loaded."
    persist_current_state()
    return True


def generate_explanation(section: StudySection) -> str:
    text = section_context(section)
    language = current_language()
    concepts = ", ".join(section.key_concepts[:4]) or ("הרעיונות המרכזיים" if language == "he" else "the main ideas")

    prompt = (
        "Explain this study section for a student preparing for an exam.\n"
        f"{tutor_language_instruction(language)}\n"
        "Use the provided section text only.\n"
        "Structure the answer with these headings:\n"
        "Summary, Key Ideas, Important Definitions, Exam Tips.\n"
        "Keep it clear and practical.\n\n"
        f"Section title: {section.title}\n"
        f"Pages: {section.page_label}\n"
        f"Key concepts: {concepts}\n\n"
        f"Section text:\n{text[:6000]}"
    )

    response = GeneralAIService().ask([], prompt, language=language)
    if response["ok"]:
        provider = response.get("provider", "AI")
        return f"{response['answer']}\n\n_AI provider: {provider}_\n\n{source_label(section)}"

    sentences = [item.strip() for item in text.replace("\n", " ").split(".") if len(item.split()) >= 6]
    summary = " ".join(sentence + "." for sentence in sentences[:2]) or section.summary
    definitions = section.key_concepts[:3] or [t("fallback_core_idea"), t("fallback_example"), t("fallback_review_point")]

    if language == "he":
        return (
            f"**סיכום**\n\n{summary}\n\n"
            f"**רעיונות מרכזיים**\n\n- {concepts}\n- קשרו את הדוגמאות ב{page_label(section)} לכותרת החלק.\n\n"
            f"**הגדרות חשובות**\n\n"
            + "\n".join(f"- {term}: הגדירו את המונח מתוך הערות החלק." for term in definitions)
            + "\n\n**טיפים למבחן**\n\n"
            "- ודאו שאתם יכולים להסביר את החלק במילים שלכם.\n"
            "- תרגלו דוגמה אחת בלי להסתכל ב-PDF.\n"
            "- חזרו על כל מושג מרכזי שאינכם יכולים להגדיר במהירות.\n\n"
            f"_הסבר גיבוי לא מקוון_\n\n{source_label(section)}"
        )

    return (
        f"**Summary**\n\n{summary}\n\n"
        f"**Key Ideas**\n\n- {concepts}\n- Connect the examples on {section.page_label.lower()} to the section title.\n\n"
        f"**Important Definitions**\n\n"
        + "\n".join(f"- {term}: define this term from the section notes." for term in definitions)
        + "\n\n**Exam Tips**\n\n"
        "- Be ready to explain the section in your own words.\n"
        "- Practice one example without looking at the PDF.\n"
        "- Review any key concept tag you cannot define quickly.\n\n"
        f"_Offline fallback explanation_\n\n{source_label(section)}"
    )


def answer_section_question(section: StudySection, question: str) -> str:
    chunks = retrieve_pdf_chunks(question)
    if not chunks:
        return NOT_ENOUGH_INFORMATION

    return answer_from_retrieved_chunks(question, chunks)


def retrieve_pdf_chunks(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not has_pdf():
        return []
    chunks = ContextRetrievalService.build_chunks_from_pages(
        st.session_state.pages,
        st.session_state.sections,
    )
    return ContextRetrievalService.retrieve_relevant_chunks(question, chunks, top_k=top_k, min_score=1)


def retrieve_ai_tutor_pdf_chunks(question: str, top_k: int = 8) -> list[dict[str, Any]]:
    if not has_pdf():
        return []

    chunks = ContextRetrievalService.build_chunks_from_pages(
        st.session_state.pages,
        st.session_state.sections,
    )
    retrieved = ContextRetrievalService.retrieve_relevant_chunks(question, chunks, top_k=top_k, min_score=1)
    if retrieved:
        return retrieved

    if is_document_overview_question(question):
        return ContextRetrievalService.retrieve_overview_chunks(chunks, top_k=top_k)

    return []


def is_document_overview_question(question: str) -> bool:
    normalized = (question or "").lower()
    compact = re.sub(r"\s+", " ", normalized).strip()
    overview_phrases = [
        "main idea",
        "main ideas",
        "main point",
        "main points",
        "key idea",
        "key ideas",
        "important idea",
        "important ideas",
        "big picture",
        "overview",
        "summarize",
        "summary",
        "what is this pdf about",
        "what is the pdf about",
        "from the pdf",
        "from my material",
        "study plan",
        "practice question",
        "practice questions",
        "quiz me",
        "prepare for",
        "help me study",
        "רעיונות מרכזיים",
        "סכם",
        "סיכום",
        "מה הנושאים",
        "מה הרעיונות",
        "מהמסמך",
        "מהחומר",
        "תוכנית לימוד",
        "שאלות תרגול",
    ]
    return any(phrase in compact for phrase in overview_phrases)


def question_mentions_pdf_context(question: str) -> bool:
    compact = re.sub(r"\s+", " ", (question or "").lower()).strip()
    pdf_phrases = [
        "pdf",
        "the document",
        "this document",
        "uploaded document",
        "uploaded file",
        "uploaded material",
        "my material",
        "this material",
        "the material",
        "from the file",
        "from the document",
        "from the pdf",
        "in the file",
        "in the document",
        "in the pdf",
        "ה-pdf",
        "המסמך",
        "בקובץ",
        "במסמך",
        "בחומר",
        "מהקובץ",
        "מהמסמך",
        "מהחומר",
    ]
    return any(phrase in compact for phrase in pdf_phrases)


def pdf_context_unavailable_message(language: str) -> str:
    if language == "he":
        return (
            "אין כרגע הקשר PDF פעיל לשאלה הזאת. עברו לעמוד Upload והעלו PDF, "
            "או לחצו Continue על סשן שמור, ואז שאלו שוב עם Use uploaded PDF context מסומן."
        )
    return (
        "There is no active PDF context loaded for this question. Go to Upload and upload a PDF, "
        "or click Continue on a saved session, then ask again with Use uploaded PDF context checked."
    )


def answer_from_retrieved_chunks(question: str, chunks: list[dict[str, Any]]) -> str:
    return answer_from_retrieved_chunks_result(question, chunks)["answer"]


def answer_from_retrieved_chunks_result(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved_context = ContextRetrievalService.format_chunks_for_prompt(chunks, max_chars=7000)
    if not retrieved_context:
        return {"ok": False, "answer": NOT_ENOUGH_INFORMATION, "provider": "local", "context": "pdf"}

    prompt = build_grounded_pdf_prompt(question, retrieved_context)
    response = GeneralAIService().complete(
        GROUNDED_SYSTEM_PROMPT,
        prompt,
        language=current_language(),
    )
    if response["ok"]:
        return {
            "ok": True,
            "answer": with_retrieved_sources(response["answer"], chunks),
            "provider": response.get("provider", "ai"),
            "context": "pdf",
        }

    if is_document_overview_question(question):
        return {
            "ok": True,
            "answer": with_retrieved_sources(local_overview_answer(question, chunks), chunks),
            "provider": "local",
            "context": "pdf",
        }

    return {
        "ok": True,
        "answer": with_retrieved_sources(local_grounded_answer(question, chunks), chunks),
        "provider": "local",
        "context": "pdf",
    }


def build_grounded_pdf_prompt(question: str, retrieved_context: str) -> str:
    return (
        "You are a grounded PDF study assistant.\n"
        "Answer ONLY using the provided PDF context.\n"
        "Do not use outside knowledge.\n"
        "Do not guess.\n"
        "If the answer is not supported by the provided PDF context, reply exactly:\n"
        f"\"{NOT_ENOUGH_INFORMATION}\"\n"
        "For document-wide questions such as summaries, main ideas, study plans, or practice questions, "
        "synthesize only across the provided context sections.\n"
        "If the user asks for a number of ideas or questions, return that number when the context supports it.\n"
        "When you answer, include a short source line using the provided section/page metadata.\n\n"
        f"Provided PDF context:\n{retrieved_context}\n\n"
        f"Question:\n{question}"
    )


def with_retrieved_sources(answer: str, chunks: list[dict[str, Any]]) -> str:
    if (answer or "").strip() == NOT_ENOUGH_INFORMATION:
        return NOT_ENOUGH_INFORMATION
    labels = ContextRetrievalService.source_labels(chunks)
    if not labels:
        return answer.strip() or NOT_ENOUGH_INFORMATION
    sources = "\n".join(f"- {label}" for label in labels)
    return f"{(answer or NOT_ENOUGH_INFORMATION).strip()}\n\nRetrieved sources:\n{sources}"


def local_grounded_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    query_tokens = ContextRetrievalService._tokens(question)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for chunk in chunks:
        text = str(chunk.get("text", "") or "")
        for sentence in split_sentences(text):
            overlap = len(query_tokens & ContextRetrievalService._tokens(sentence))
            if overlap:
                candidates.append((overlap, sentence, chunk))

    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    if candidates:
        _, sentence, chunk = candidates[0]
        return f"{sentence}\n\n{source_line(chunk)}"

    first_chunk = chunks[0] if chunks else {}
    excerpt = " ".join(str(first_chunk.get("text", "") or "").split())[:500].rstrip()
    if not excerpt:
        return NOT_ENOUGH_INFORMATION
    return f"{excerpt}\n\n{source_line(first_chunk)}"


def local_overview_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    requested_count = requested_item_count(question, default=5)
    ideas: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        title = str(chunk.get("section_title", "") or "Study section").strip()
        text = " ".join(str(chunk.get("text", "") or "").split())
        sentence = split_sentences(text)[:1]
        detail = sentence[0] if sentence else text[:180].rstrip()
        if not detail:
            continue
        idea = f"{title}: {detail}"
        key = idea.lower()
        if key in seen:
            continue
        seen.add(key)
        ideas.append(idea)
        if len(ideas) >= requested_count:
            break

    if not ideas:
        return NOT_ENOUGH_INFORMATION

    heading = "Main ideas from the uploaded PDF:"
    if current_language() == "he":
        heading = "הרעיונות המרכזיים מה-PDF שהועלה:"
    return heading + "\n\n" + "\n".join(f"{index}. {idea}" for index, idea in enumerate(ideas, start=1))


def requested_item_count(question: str, default: int = 5) -> int:
    match = re.search(r"\b([1-9]|10)\b", question or "")
    if not match:
        return default
    return max(1, min(10, int(match.group(1))))


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or "")
        if len(sentence.strip().split()) >= 5
    ]


def source_line(chunk: dict[str, Any]) -> str:
    section_number = int(chunk.get("section_number", 0) or 0)
    title = str(chunk.get("section_title", "") or "Untitled section")
    start_page = int(chunk.get("start_page", chunk.get("page", 0)) or 0)
    end_page = int(chunk.get("end_page", start_page) or start_page)
    if start_page == end_page:
        return f"Source: Section {section_number} - {title} - Page {start_page}"
    return f"Source: Section {section_number} - {title} - Pages {start_page}-{end_page}"


def answer_ai_tutor(question: str, use_pdf_context: bool = False) -> dict[str, Any]:
    language = current_language()
    active_pdf = has_pdf()
    mentions_pdf_context = question_mentions_pdf_context(question)
    wants_pdf_context = use_pdf_context or mentions_pdf_context or is_document_overview_question(question)

    if mentions_pdf_context and not active_pdf:
        return {"ok": False, "answer": pdf_context_unavailable_message(language), "provider": "local"}

    if wants_pdf_context and active_pdf:
        chunks = retrieve_ai_tutor_pdf_chunks(question)
        if not chunks:
            return {"ok": False, "answer": NOT_ENOUGH_INFORMATION, "provider": "local"}
        return answer_from_retrieved_chunks_result(question, chunks)

    messages = list(st.session_state.ai_tutor_history)
    result = GeneralAIService().ask(messages, question, language=language)
    if result["ok"]:
        result["answer"] = (
            f"{result['answer']}\n\n"
            "_General AI mode: this answer is not grounded in the uploaded PDF._"
        )
        return result

    if language == "he":
        available = "אפשר להשתמש בחלקים מה-PDF שהעלית." if has_pdf() else "העלו PDF כדי לתת לי הקשר לימודי."
        fallback = (
            "מורה ה-AI צריך `OPENAI_API_KEY` או `GROQ_API_KEY` כדי לענות תשובה מלאה. "
            f"{available} בינתיים נסו לשאול שאלה ממוקדת כמו "
            "\"סכם את החלק הזה\", \"בחן אותי על המושגים המרכזיים\", או "
            "\"הסבר את המונח הקשה ביותר בפשטות.\""
        )
    else:
        available = "I can use your uploaded PDF sections." if has_pdf() else "Upload a PDF to give me study context."
        fallback = (
            "AI Tutor needs `OPENAI_API_KEY` or `GROQ_API_KEY` for a full general-mode answer. "
            f"{available} For now, try asking one focused question such as "
            "\"summarize this section\", \"quiz me on the key concepts\", or "
            "\"explain the hardest term in simple words.\""
        )
    return {"ok": False, "answer": fallback, "provider": "none"}


def build_section_quiz(section: StudySection) -> list[dict[str, Any]]:
    generated = QuizService.generate_from_documents(
        [{"text": section_context(section), "source": st.session_state.pdf_name, "page": section.start_page}],
        num_questions=3,
        language=current_language(),
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
            "question": (
                f"נכון או לא נכון: {concept} מופיע בחלק הזה."
                if current_language() == "he"
                else f"True or False: {concept} is discussed in this section."
            ),
            "options": [t("true"), t("false")],
            "answer": t("true"),
            "source_page": section.start_page,
        }
    )
    questions.append(
        {
            "type": "short_answer",
            "question": (
                f"במשפט אחד, הסבירו למה {concept} חשוב בחלק הזה."
                if current_language() == "he"
                else f"In one sentence, explain why {concept} matters in this section."
            ),
            "options": [],
            "answer": (
                "תשובה טובה צריכה להשתמש בטקסט החלק ולהזכיר בבירור את הרעיון המרכזי."
                if current_language() == "he"
                else "A strong answer should use the section text and mention the main idea clearly."
            ),
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
    context = ContextRetrievalService.retrieve_exam_context(
        st.session_state.sections,
        st.session_state.pages,
        max_chars=12000,
    )
    return ExamService().generate_final_exam(
        context,
        ExamOptions(question_count=int(question_count), difficulty=difficulty, language=current_language()),
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
        return t("weak_review_complete")

    quiz_average = ProgressService.quiz_average(st.session_state.progress)
    plans = [f"### {t('weak_review_title')}"]
    for title in review_sections:
        section = next((item for item in st.session_state.sections if item.title == title), None)
        if section is None:
            continue
        topic = section.key_concepts[0] if section.key_concepts else section.title
        reason = t("reason_low_quiz") if quiz_average and quiz_average < 80 else t("reason_not_completed")
        plans.append(
            "- **"
            + t(
                "weak_review_item",
                topic=html.escape(topic),
                reason=reason,
                number=section.section_number,
                page_label=page_label(section).lower(),
            )
            + "**"
        )
    return "\n".join(plans)
