from __future__ import annotations

import html
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.exam_service import ExamOptions, ExamService
from services.general_ai_service import GeneralAIService
from services.pdf_render_service import PdfRenderService
from services.pdf_section_service import PdfSectionError, PdfSectionService
from services.pdf_service import PdfExtractionError, PdfService
from services.progress_service import ProgressService
from services.quiz_service import QuizService
from services.study_service import StudySection, StudyService
from ui.navigation import DEFAULT_CURRENT_PAGE, NAV_ITEMS, normalize_current_page


st.set_page_config(page_title="Smart Study Assistant", page_icon="📘", layout="wide")


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F8FAFC; color: #0F172A; }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #0F172A; letter-spacing: 0; }
        [data-testid="stSidebar"] { display: none; }
        .top-nav {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: .75rem .9rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .07);
            position: sticky;
            top: 1rem;
            z-index: 20;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: .45rem;
            min-height: 2.4rem;
            color: #1E3A8A;
            font-weight: 800;
            font-size: 1.05rem;
            white-space: nowrap;
        }
        .nav-active {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.45rem;
            border-radius: 999px;
            background: #DBEAFE;
            color: #1E3A8A;
            font-weight: 800;
            border: 1px solid #BFDBFE;
        }
        .status-bar {
            display: flex;
            flex-wrap: wrap;
            gap: .75rem;
            align-items: center;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: .7rem .95rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, .045);
            color: #64748B;
            font-size: .9rem;
        }
        .status-bar strong { color: #0F172A; }
        .hero-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
            color: #FFFFFF;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 34px rgba(30, 58, 138, .16);
        }
        .hero-card h1 { color: #FFFFFF; margin: 0; font-size: 1.55rem; }
        .hero-card p { color: #E0F2FE; margin: .35rem 0 0; }
        .card {
            background: #ffffff;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.05rem 1.1rem;
            margin-bottom: .85rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .07);
        }
        .card-title { color: #1E3A8A; font-weight: 700; font-size: 1.04rem; margin-bottom: .35rem; }
        .muted { color: #64748B; font-size: .9rem; }
        .roadmap-card {
            border-left: 5px solid #0EA5A4;
            position: relative;
        }
        .roadmap-index {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #1E3A8A;
            color: white;
            font-weight: 800;
            margin-right: .45rem;
        }
        .objective-list { margin: .45rem 0 .2rem 1.15rem; color: #0F172A; }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: .18rem .55rem;
            font-weight: 700;
            font-size: .75rem;
            margin-right: .25rem;
        }
        .badge-primary { background: #DBEAFE; color: #1E3A8A; }
        .badge-accent { background: #CCFBF1; color: #0F766E; }
        .badge-success { background: #DCFCE7; color: #16A34A; }
        .badge-warning { background: #FEF3C7; color: #B45309; }
        .tag {
            display: inline-block;
            background: #CCFBF1;
            color: #0F766E;
            border-radius: 999px;
            padding: .16rem .48rem;
            margin: .16rem .18rem .16rem 0;
            font-size: .75rem;
        }
        .prompt-card {
            background: #FFFFFF;
            border: 1px solid #DDE7F0;
            border-radius: 12px;
            padding: .8rem .9rem;
            min-height: 4.2rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, .05);
        }
        .source-label { color: #64748B; font-size: .85rem; }
        div.stButton > button {
            border-radius: 999px;
            border-color: #E2E8F0;
            color: #0F172A;
        }
        div.stButton > button[kind="primary"] { background: #1E3A8A; border-color: #1E3A8A; }
        div.stButton > button:hover { border-color: #0EA5A4; color: #0F172A; }
        div[data-testid="stProgress"] > div > div > div { background-color: #0EA5A4; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "pdf_bytes": b"",
        "pdf_name": "",
        "pages": [],
        "pending_pages": [],
        "pending_sections": [],
        "pending_pdf_bytes": b"",
        "pending_pdf_name": "",
        "sections": [],
        "current_section_index": 0,
        "upload_message": "",
        "section_explanation": "",
        "section_quiz": [],
        "section_quiz_answers": {},
        "section_quiz_score": None,
        "section_answer": "",
        "ask_ai_history": [],
        "final_exam": None,
        "weak_topic_review": "",
        "current_page": DEFAULT_CURRENT_PAGE,
        "progress": ProgressService.default_state(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.progress = ProgressService.load(st.session_state.progress)
    st.session_state.current_page = normalize_current_page(st.session_state.get("current_page"))
    st.session_state.sections = normalize_study_sections(st.session_state.sections)
    st.session_state.pending_sections = normalize_study_sections(st.session_state.pending_sections)


def normalize_study_section(section: Any) -> StudySection:
    if isinstance(section, StudySection):
        return section

    if isinstance(section, dict):
        data = section
    elif hasattr(section, "__dict__"):
        data = vars(section)
    else:
        data = {}

    return StudySection(
        section_number=int(data.get("section_number", 0)),
        title=str(data.get("title", "")),
        start_page=int(data.get("start_page", 0)),
        end_page=int(data.get("end_page", 0)),
        estimated_minutes=int(data.get("estimated_minutes", 0)),
        difficulty=str(data.get("difficulty", "Easy")),
        summary=str(data.get("summary", "")),
        learning_objectives=list(data.get("learning_objectives") or []),
        key_concepts=list(data.get("key_concepts") or []),
    )


def normalize_study_sections(sections: list[Any]) -> list[StudySection]:
    if not sections:
        return []
    return [normalize_study_section(section) for section in sections]


def card(title: str, body: str, extra: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{html.escape(title)}</div>
            <div>{body}</div>
            {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


def roadmap_card(section: StudySection, body: str) -> None:
    st.markdown(
        f"""
        <div class="card roadmap-card">
            <div class="card-title"><span class="roadmap-index">{section.section_number}</span>{html.escape(section.title)}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "primary") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def format_seconds(seconds: int) -> str:
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {remainder}s"


def has_pdf() -> bool:
    return bool(st.session_state.pdf_bytes and st.session_state.pages and st.session_state.sections)


def current_section() -> StudySection | None:
    sections = st.session_state.sections
    if not sections:
        return None
    index = min(max(0, st.session_state.current_section_index), len(sections) - 1)
    st.session_state.current_section_index = index
    return sections[index]


def reset_section_outputs() -> None:
    st.session_state.section_explanation = ""
    st.session_state.section_quiz = []
    st.session_state.section_quiz_answers = {}
    st.session_state.section_quiz_score = None
    st.session_state.section_answer = ""


def source_label(section: StudySection, page: int | None = None) -> str:
    if page is not None:
        return f"Source: Section {section.section_number}, Page {page}"
    if section.start_page == section.end_page:
        return f"Source: Page {section.start_page}"
    return f"Source: Pages {section.start_page}-{section.end_page}"


def extract_pdf(uploaded_file: Any) -> None:
    pdf_bytes = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pages = PdfService().extract_pages(tmp.name)

    st.session_state.pending_pdf_bytes = pdf_bytes
    st.session_state.pending_pdf_name = uploaded_file.name
    st.session_state.pending_pages = pages
    st.session_state.pending_sections = StudyService().generate_study_plan(pages)
    if not st.session_state.pending_sections:
        raise PdfExtractionError("No readable study sections could be created from this PDF.")
    st.session_state.upload_message = f"Processed {uploaded_file.name}. Ready to generate a study plan."


def generate_study_plan_from_pending() -> None:
    pages = st.session_state.pending_pages
    sections = st.session_state.pending_sections or StudyService().generate_study_plan(pages)
    if not sections:
        raise PdfExtractionError("No readable study sections could be created from this PDF.")

    st.session_state.pdf_bytes = st.session_state.pending_pdf_bytes
    st.session_state.pdf_name = st.session_state.pending_pdf_name
    st.session_state.pages = pages
    st.session_state.sections = sections
    st.session_state.current_section_index = 0
    st.session_state.upload_message = f"Generated {len(sections)} study sections."
    st.session_state.progress = ProgressService.default_state()
    st.session_state.final_exam = None
    reset_section_outputs()


def section_context(section: StudySection) -> str:
    return StudyService.section_text(st.session_state.pages, section)


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
        [
            {
                "role": "user",
                "content": f"Use this study section as context when helpful:\n{text[:5000]}",
            }
        ],
        question,
    )
    if response["ok"]:
        return f"{response['answer']}\n\n{source_label(section)}"
    sentences = [item.strip() for item in text.replace("\n", " ").split(".") if len(item.split()) > 6]
    answer = (sentences[0] + ".") if sentences else response["answer"]
    return f"{answer}\n\n{source_label(section)}"


def render_top_nav() -> None:
    with st.container(border=True):
        st.markdown('<div class="top-nav">', unsafe_allow_html=True)
        columns = st.columns([2.25, 1, 1.15, 1.15, 1, 1, 1])
        with columns[0]:
            st.markdown('<div class="brand">📘 Smart Study Assistant</div>', unsafe_allow_html=True)
        for index, label in enumerate(NAV_ITEMS, start=1):
            with columns[index]:
                if st.session_state.current_page == label:
                    st.markdown(f'<div class="nav-active">{html.escape(label)}</div>', unsafe_allow_html=True)
                elif st.button(label, key=f"nav-{label}", use_container_width=True):
                    st.session_state.current_page = label
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def render_status_bar() -> None:
    if has_pdf():
        status = (
            f"<span><strong>Current PDF:</strong> {html.escape(st.session_state.pdf_name)}</span>"
            f"<span><strong>Pages:</strong> {len(st.session_state.pages)}</span>"
            f"<span><strong>Sections:</strong> {len(st.session_state.sections)}</span>"
            f"<span><strong>Progress:</strong> {overall_progress():.0f}%</span>"
        )
    else:
        status = "No PDF loaded yet — upload a document to begin."
    st.markdown(f'<div class="status-bar">{status}</div>', unsafe_allow_html=True)


def render_upload_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Turn PDFs into guided study sessions, quizzes, and exam practice.</h1>
            <p>Upload course material, study section by section, and ask the AI tutor for help.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload() -> None:
    render_upload_hero()
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Process PDF", type="primary", disabled=uploaded_file is None):
        try:
            with st.spinner("Extracting text from your PDF..."):
                extract_pdf(uploaded_file)
            st.success(st.session_state.upload_message)
        except PdfExtractionError as exc:
            st.error(f"PDF extraction failed: {exc}")
        except Exception as exc:
            st.error(f"Could not process PDF safely: {exc}")

    if st.session_state.pending_pages:
        draft_sections = st.session_state.pending_sections or st.session_state.sections
        estimated = sum(section.estimated_minutes for section in draft_sections)
        cols = st.columns(3)
        cols[0].metric("Pages processed", len(st.session_state.pending_pages))
        cols[1].metric("Total sections generated", len(draft_sections))
        cols[2].metric("Estimated study time", f"{estimated} min")
        if st.button("Generate Study Plan", type="primary"):
            try:
                generate_study_plan_from_pending()
                st.success(st.session_state.upload_message)
                st.session_state.current_page = "Study Plan"
                st.rerun()
            except PdfExtractionError as exc:
                st.error(str(exc))


def section_status(section: StudySection) -> str:
    if section.section_number in st.session_state.progress.completed_sections:
        return "Completed"
    current = current_section()
    if current and current.section_number == section.section_number:
        return "In Progress"
    return "Not Started"


def render_study_plan() -> None:
    st.subheader("Study Plan")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    total_time = sum(section.estimated_minutes for section in st.session_state.sections)
    cols = st.columns(3)
    cols[0].metric("Total sections", len(st.session_state.sections))
    cols[1].metric("Total estimated time", f"{total_time} min")
    cols[2].metric("Completed sections", len(st.session_state.progress.completed_sections))
    st.progress(overall_progress() / 100)
    st.caption(f"{overall_progress():.0f}%")

    for index, section in enumerate(st.session_state.sections):
        status = section_status(section)
        status_kind = "success" if status == "Completed" else "accent" if status == "In Progress" else "warning"
        concepts = "".join(
            f'<span class="tag">{html.escape(tag)}</span>' for tag in getattr(section, "key_concepts", [])
        )
        objectives = "".join(
            f"<li>{html.escape(objective)}</li>" for objective in getattr(section, "learning_objectives", [])[:5]
        )
        body = (
            f"{badge(section.page_label, 'accent')}"
            f"{badge(section.difficulty, 'warning')}"
            f"{badge(status, status_kind)}"
            f"<p>{html.escape(section.summary)}</p>"
            f"<div class='muted'><strong>Learning objectives</strong></div>"
            f"<ul class='objective-list'>{objectives}</ul>"
            f"<div>{concepts}</div>"
            f"<p class='muted'>Estimated time: {section.estimated_minutes} minutes</p>"
        )
        roadmap_card(section, body)
        if st.button("Start Studying", key=f"start-section-{section.section_number}"):
            st.session_state.current_section_index = index
            reset_section_outputs()
            st.session_state.current_page = "Study Mode"
            st.rerun()


def render_pdf_pages(section: StudySection) -> None:
    st.markdown(f"### PDF Section")
    st.markdown(f"**Now studying {section.page_label}**")
    images = PdfRenderService.render_pages(st.session_state.pdf_bytes, section.start_page, section.end_page)
    if images:
        for offset, image in enumerate(images, start=section.start_page):
            st.image(image, caption=source_label(section, offset), use_container_width=True)
    else:
        st.info("Page images are unavailable. Use the extracted text fallback below.")

    try:
        section_pdf = PdfSectionService.extract_section_pdf(
            st.session_state.pdf_bytes,
            section.start_page,
            section.end_page,
        )
        st.download_button(
            "Download section PDF",
            data=section_pdf,
            file_name=f"section-{section.section_number}.pdf",
            mime="application/pdf",
        )
    except PdfSectionError:
        st.caption("Section PDF download is unavailable for this page range.")

    with st.expander("Extracted text fallback"):
        st.text_area("Section text", value=section_context(section), height=300, label_visibility="collapsed")


def render_study_mode() -> None:
    st.subheader("Study Mode")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    section = current_section()
    if section is None:
        st.info("Generate a study plan first.")
        return

    left, right = st.columns([0.65, 0.35])
    with left:
        render_pdf_pages(section)

    with right:
        card(
            section.title,
            f"{badge(f'Section {section.section_number} of {len(st.session_state.sections)}', 'primary')}"
            f"{badge(section.page_label, 'accent')}"
            f"{badge(section.difficulty, 'warning')}"
            f"<p class='muted'>Estimated: {section.estimated_minutes} minutes</p>",
        )
        st.progress((section.section_number - 1) / max(1, len(st.session_state.sections)))
        st.metric("Actual study time", format_seconds(st.session_state.progress.actual_study_seconds))
        timer_cols = st.columns(3)
        if timer_cols[0].button("Start Session"):
            st.session_state.progress = ProgressService.start_timer(st.session_state.progress)
            st.rerun()
        if timer_cols[1].button("Pause"):
            st.session_state.progress = ProgressService.pause_timer(st.session_state.progress)
            st.rerun()
        if timer_cols[2].button("Finish Section", type="primary"):
            st.session_state.progress = ProgressService.finish_section(
                st.session_state.progress,
                section.section_number,
            )
            st.rerun()

        if st.button("Explain This Section", use_container_width=True):
            st.session_state.section_explanation = generate_explanation(section)
        if st.session_state.section_explanation:
            with st.expander("Explanation", expanded=True):
                st.markdown(st.session_state.section_explanation)

        with st.expander("Quiz", expanded=bool(st.session_state.section_quiz)):
            if st.button("Generate Quiz", use_container_width=True):
                st.session_state.section_quiz = build_section_quiz(section)
                st.session_state.section_quiz_answers = {}
                st.session_state.section_quiz_score = None
            if st.session_state.section_quiz:
                render_section_quiz(section)

        with st.expander("Ask a question about this section"):
            question = st.text_input("Question", key=f"section-question-{section.section_number}")
            if st.button("Ask About This Section", disabled=not question.strip()):
                st.session_state.section_answer = answer_section_question(section, question)
            if st.session_state.section_answer:
                st.markdown(st.session_state.section_answer)

        if st.button("Next Section", use_container_width=True):
            st.session_state.current_section_index = StudyService.next_section_index(
                st.session_state.current_section_index,
                len(st.session_state.sections),
            )
            reset_section_outputs()
            st.rerun()


def build_section_quiz(section: StudySection) -> list[dict[str, Any]]:
    generated = QuizService.generate_from_documents(
        [{"text": section_context(section), "source": st.session_state.pdf_name, "page": section.start_page}],
        num_questions=3,
    )
    questions: list[dict[str, Any]] = []
    if generated:
        first = generated[0]
        questions.append(
            {
                "type": "multiple_choice",
                "question": first.prompt,
                "options": first.options,
                "answer": first.answer,
                "source_page": first.page or section.start_page,
            }
        )
    text = section_context(section)
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


def render_section_quiz(section: StudySection) -> None:
    correct = 0
    scorable = 0
    for index, question in enumerate(st.session_state.section_quiz, start=1):
        st.markdown(f"**{index}. {question['question']}**")
        key = f"quiz-answer-{section.section_number}-{index}"
        if question["type"] in {"multiple_choice", "true_false"}:
            answer = st.radio("Answer", question["options"], key=key, label_visibility="collapsed")
            st.session_state.section_quiz_answers[index] = answer
            scorable += 1
            if answer == question["answer"]:
                correct += 1
        else:
            answer = st.text_area("Short answer", key=key, height=80)
            st.session_state.section_quiz_answers[index] = answer
        st.caption(source_label(section, question.get("source_page")))

    if st.button("Submit quiz"):
        score = round((correct / scorable) * 100) if scorable else 100
        st.session_state.section_quiz_score = score
        st.session_state.progress.quiz_scores.append(float(score))
    if st.session_state.section_quiz_score is not None:
        st.success(f"Score: {st.session_state.section_quiz_score}%")
        with st.expander("Review answers"):
            for index, question in enumerate(st.session_state.section_quiz, start=1):
                st.write(f"{index}. Correct answer: {question['answer']}")


def render_ask_ai() -> None:
    st.subheader("AI Tutor")
    st.caption("Ask general questions, request examples, or get help understanding a topic.")
    prompts = [
        "Explain recursion with an example",
        "Help me prepare for an algorithms exam",
        "Explain Big O notation simply",
        "Create practice questions",
    ]
    prompt_cols = st.columns(4)
    for index, prompt_text in enumerate(prompts):
        with prompt_cols[index]:
            st.markdown(f"<div class='prompt-card'>{html.escape(prompt_text)}</div>", unsafe_allow_html=True)

    if st.button("Clear Chat"):
        st.session_state.ask_ai_history = []
        st.rerun()

    for message in st.session_state.ask_ai_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a general study question")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        result = GeneralAIService().ask(st.session_state.ask_ai_history, prompt)
        with st.chat_message("assistant"):
            st.write(result["answer"])
            if result["provider"] == "none":
                st.caption("Setup needed: add OPENAI_API_KEY or GROQ_API_KEY.")
            else:
                st.caption(f"Provider: {result['provider']}")
        st.session_state.ask_ai_history.extend(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": result["answer"]}]
        )


def all_study_context() -> str:
    lines: list[str] = []
    for section in st.session_state.sections:
        lines.append(f"{section.title}\n{section_context(section)}")
    return "\n\n".join(lines)


def render_final_exam() -> None:
    st.subheader("Final Exam")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    cols = st.columns(2)
    question_count = cols[0].number_input("Questions", min_value=3, max_value=25, value=10)
    difficulty = cols[1].selectbox("Difficulty", ["mixed", "easy", "medium", "hard"])
    if st.button("Generate AI final exam", type="primary"):
        with st.spinner("Generating final exam..."):
            st.session_state.final_exam = ExamService().generate_final_exam(
                all_study_context(),
                ExamOptions(question_count=int(question_count), difficulty=difficulty),
            )

    exam = st.session_state.final_exam
    if not exam:
        return
    if exam.get("fallback_used"):
        st.warning(exam.get("fallback_note", "Fallback exam was used."))
    st.markdown(f"### {exam.get('title', 'AI Final Exam')}")
    for question in exam.get("questions", []):
        with st.expander(f"{question.get('id')}. {question.get('question')}"):
            for option in question.get("options", []):
                st.write(f"- {option}")
            st.success(f"Answer: {question.get('answer')}")
            st.caption(f"Topic: {question.get('topic', 'General')}")
    score = st.slider("Score", min_value=0, max_value=100, value=85)
    if st.button("Save final exam score"):
        st.session_state.progress.final_exam_score = float(score)
        st.success("Final exam score saved.")
    correct = round(len(exam.get("questions", [])) * score / 100)
    wrong = max(0, len(exam.get("questions", [])) - correct)
    cols = st.columns(3)
    cols[0].metric("Score", f"{score}%")
    cols[1].metric("Correct answers", correct)
    cols[2].metric("Wrong answers", wrong)
    review = recommended_review_sections()
    if review:
        card("Suggested review sections", html.escape(", ".join(review)))


def overall_progress() -> float:
    total = len(st.session_state.sections)
    if total == 0:
        return 0.0
    return len(st.session_state.progress.completed_sections) / total * 100


def render_dashboard() -> None:
    st.subheader("Dashboard")
    if not has_pdf():
        st.info("Upload and process a PDF first.")
        return

    progress = st.session_state.progress
    cols = st.columns(4)
    cols[0].metric("Learning Progress", f"{overall_progress():.0f}%")
    cols[1].metric("Completed Sections", f"{len(progress.completed_sections)}/{len(st.session_state.sections)}")
    cols[2].metric("Quiz Average", f"{ProgressService.quiz_average(progress):.0f}%")
    readiness = progress.final_exam_score if progress.final_exam_score is not None else overall_progress()
    cols[3].metric("Exam Readiness", f"{readiness:.0f}%")
    st.progress(overall_progress() / 100)

    total_sections = max(1, len(st.session_state.sections))
    average_seconds = progress.actual_study_seconds // total_sections
    with st.expander("Study time"):
        cols = st.columns(2)
        cols[0].metric("Total study time", format_seconds(progress.actual_study_seconds))
        cols[1].metric("Average per section", format_seconds(average_seconds))

    review = recommended_review_sections()
    next_section = next_recommended_section()
    recommendation = []
    if review:
        recommendation.append(f"Review {review[0]} before taking the final exam again.")
    if next_section:
        recommendation.append(f"Recommended next section: {next_section.title}.")
    card("Recommendations", html.escape(" ".join(recommendation) if recommendation else "Keep reviewing completed sections."))

    if st.button("Review Weak Topics", type="primary"):
        st.session_state.weak_topic_review = build_weak_topic_review()
    if st.session_state.weak_topic_review:
        st.markdown(st.session_state.weak_topic_review)


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
            f"- **Review {html.escape(topic)}** — {reason} in Section {section.section_number}. "
            f"Re-read {section.page_label.lower()} and retake the section quiz."
        )
    return "\n".join(plans)


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


PAGE_RENDERERS = {
    "Upload": render_upload,
    "Study Plan": render_study_plan,
    "Study Mode": render_study_mode,
    "AI Tutor": render_ask_ai,
    "Final Exam": render_final_exam,
    "Dashboard": render_dashboard,
}


inject_custom_css()
init_state()
render_top_nav()
render_status_bar()

PAGE_RENDERERS[st.session_state.current_page]()
