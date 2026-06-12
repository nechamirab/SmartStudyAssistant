# Smart Study Assistant Proposal Outline

## Repository Findings

### What the Current Repository Implements

- **Project identity:** `Smart Study Assistant`, a student-facing Streamlit MVP for learning from uploaded PDF documents.
- **Main product flow:** `Upload PDF -> Study Plan -> Study Mode -> AI Tutor -> Final Exam -> Dashboard`.
- **Main app entry point:** `ui/streamlit_app.py`.
- **Page modules:** `pages/upload_page.py`, `pages/study_plan_page.py`, `pages/study_mode_page.py`, `pages/ai_tutor_page.py`, `pages/final_exam_page.py`, `pages/dashboard_page.py`.
- **Shared UI modules:** `ui/navigation.py`, `ui/components.py`, `ui/styles.py`, `ui/state.py`, `ui/workflow.py`.
- **PDF extraction:** `services/pdf_service.py` extracts text with PyMuPDF first and falls back to `pypdf`.
- **PDF rendering/download:** `services/pdf_render_service.py` renders selected pages to PNG bytes; `services/pdf_section_service.py` extracts a section PDF for download.
- **Study plan generation:** `services/study_service.py` creates `StudySection` objects with title, page range, estimated minutes, difficulty, summary, learning objectives, and key concepts.
- **Configurable sessions:** `StudyService.suggest_session_count(...)` suggests 3-15 sessions based mainly on word count and page count; Upload lets the user override the number before generating the plan.
- **Study Mode:** supports rendered PDF pages, extracted-text fallback, timer, per-section explanation, quiz, and question answering.
- **Per-section state:** `services/section_state_service.py` prevents section quizzes, explanations, answers, and scores from overwriting each other.
- **AI Tutor:** `services/general_ai_service.py` selects OpenAI first, Groq second, and provides clear fallback if no key exists. `pages/ai_tutor_page.py` has general tutor mode and current-section mode.
- **Quiz generation:** `services/quiz_service.py` deterministically generates fill-in-the-blank multiple-choice questions from section text; Study Mode also adds true/false and short-answer questions.
- **Quiz grading:** `services/quiz_grading_service.py` grades objective questions and accepts a short-answer evaluator supplied by the UI workflow.
- **Final exam:** `services/exam_service.py` generates a JSON final exam through OpenAI/Groq if available, with defensive parsing and fallback exam generation.
- **Interactive exam grading:** `services/exam_grading_service.py` grades final exam answers, detects weak topics/sections, and links missed answers back to study sections when possible.
- **Progress tracking:** `services/progress_service.py` tracks completed sections, quiz scores, study time, weak topics, weak sections, final exam score, and timer state.
- **Persistence:** `services/persistence_service.py` saves/restores progress locally as `.smartstudy_progress.json`.
- **Tests:** main MVP tests are in `tests/test_core.py`, `tests/test_navigation.py`, and `tests/test_mvp_recovery.py`.
- **Legacy/research code:** older RAG, LangChain, vector store, embedding, retrieval, reranking, benchmark, and OCR-related code exists under `research/`. It is no longer the main product surface.

### What Is Missing Or Should Be Described As Future Work

- OCR is not part of the main MVP flow. OCR-related code exists only in the legacy/research area, so scanned PDFs should be described as future work or preprocessing.
- The main MVP does not currently use a full vector-store RAG pipeline in the student-facing path. It uses document/section context directly, while the research folder contains RAG experimentation assets.
- FAISS, LangChain, sentence-transformers, and benchmark tooling are present only under `research/requirements.txt`, not the main MVP requirements.
- There is no authentication or multi-user cloud storage.
- Persistence is local JSON, not a database-backed user account system.
- Hebrew/RTL support is not implemented as a dedicated feature.
- Evaluation is currently unit-test and manual-flow oriented; full learning-outcome evaluation remains future work.

## Project Identity

- **Name:** Smart Study Assistant.
- **Purpose:** Help students convert course PDFs into a guided study workflow with study sessions, explanations, quizzes, AI tutoring, final exam practice, and progress tracking.
- **Target users:** Students who receive lecture notes, articles, or PDF course material and need structured study support.
- **Problem solved:** Students often have long PDF materials but no clear plan, no immediate practice questions, and no progress overview. The assistant turns static documents into an interactive learning process.
- **Workflow:** Upload PDF -> Study Plan -> Study Mode -> AI Tutor -> Final Exam -> Dashboard.

## Current Implemented Features

- Automatic PDF processing after upload.
- Text extraction with PyMuPDF and pypdf fallback.
- Suggested study session count based on extracted PDF length.
- Manual control over number of study sessions.
- Study plan generation with summaries, key concepts, learning objectives, difficulty, page ranges, and estimated time.
- Section-by-section Study Mode with rendered PDF pages and extracted text fallback.
- Per-section explanations, quizzes, and section questions.
- General AI Tutor and current-section tutor mode.
- Helpful fallback messages when no API key exists.
- Final exam generation with OpenAI/Groq and safe fallback exam generation.
- Interactive final exam answering and grading.
- Weak topic and weak section detection from missed final exam answers.
- Progress dashboard with completed sessions, quiz average, study time, exam readiness, weak topics, and recommendations.
- Local JSON progress persistence.
- Calm Streamlit UI with custom navigation and shared CSS.

## Architecture Map

### Streamlit UI Layer

- `ui/streamlit_app.py`
  - Input: Streamlit session state and navigation state.
  - Output: selected page rendering.
  - Role: app entry point, page configuration, CSS injection, navigation, status bar, page routing.

- `pages/upload_page.py`
  - Input: uploaded PDF file.
  - Output: pending extracted pages, suggested session count, generated study plan trigger.
  - Role: PDF upload, automatic extraction, session-count selection.

- `pages/study_plan_page.py`
  - Input: generated sections and progress.
  - Output: study plan cards and navigation into Study Mode.
  - Role: show current/completed/next sessions and plan metadata.

- `pages/study_mode_page.py`
  - Input: current section, PDF bytes/pages, section state, progress state.
  - Output: rendered PDF, section explanation, quiz, section answer, progress updates.
  - Role: main learning workspace.

- `pages/ai_tutor_page.py`
  - Input: general question, optional PDF context, current-section context.
  - Output: AI or fallback response.
  - Role: tutoring mode and current-section help.

- `pages/final_exam_page.py`
  - Input: study context, exam options, student answers.
  - Output: generated exam, score, weak topics/sections, review recommendations.
  - Role: exam practice and grading.

- `pages/dashboard_page.py`
  - Input: progress state and section list.
  - Output: progress metrics and recommendations.
  - Role: learning overview.

### Service Layer

- `StudyService`
  - Input: extracted `DocumentPage` list, target session count.
  - Output: list of `StudySection`.
  - Role: study plan/session generation and session count suggestion.

- `PdfService`
  - Input: local PDF path.
  - Output: list of `DocumentPage`.
  - Role: text extraction with fallback.

- `PdfRenderService` and `PdfSectionService`
  - Input: PDF bytes and page range.
  - Output: PNG page images or section PDF bytes.
  - Role: visual PDF support and section download.

- `GeneralAIService`
  - Input: messages and user question.
  - Output: response dictionary with `ok`, `answer`, and `provider`.
  - Role: OpenAI/Groq provider selection and AI Tutor calls.

- `ExamService`
  - Input: study context and exam options.
  - Output: normalized exam payload.
  - Role: AI exam generation and safe fallback handling.

- `ExamGradingService`
  - Input: exam payload, student answers, study sections.
  - Output: score, correct/wrong counts, weak topics, weak sections, results, recommendation.
  - Role: interactive final exam grading.

- `ProgressService`
  - Input: progress state events.
  - Output: updated progress state and serialized shape.
  - Role: completed sections, quiz scores, timer, final score, weak topics.

- `PersistenceService`
  - Input: app state payload.
  - Output: `.smartstudy_progress.json`.
  - Role: local save/restore.

## Technologies

- Python, with code currently running under Python 3.13 in the local environment.
- Streamlit for the web UI.
- PyMuPDF (`fitz`) and pypdf for PDF extraction/rendering.
- OpenAI API support through the `openai` Python package.
- Groq-compatible API support through `urllib.request`.
- Python `unittest` for main MVP tests.
- Local JSON persistence.
- Research-only technologies under `research/`: LangChain, FAISS, HuggingFace/sentence-transformers, vector stores, retrieval/reranking/benchmark modules, OCR-related modules.

## Updated Requirements

### Functional Requirements

- Upload a PDF.
- Extract readable text.
- Generate a study plan.
- Create study sessions.
- Let the user choose the number of study sessions.
- Suggest session number based on PDF length.
- Study section-by-section.
- Ask general AI Tutor questions.
- Ask questions about the current section/PDF.
- Generate section quizzes.
- Generate a final exam.
- Grade the final exam.
- Track progress and weak topics.
- Save and restore progress locally.

### Non-Functional Requirements

- Clear and student-friendly UI/UX.
- Fast enough response time for normal course PDFs.
- Reliable fallback behavior when APIs are missing or AI output is malformed.
- Defensive handling of invalid PDFs and empty extracted text.
- Local privacy for uploaded files and progress.
- Simple maintainable architecture with UI separated from service logic.

### Research/AI Requirements

- Use document-grounded generation and RAG principles where helpful.
- Reduce hallucinations by passing PDF/section context to AI calls.
- Evaluate answer relevance, grounding, quiz quality, and student progress.
- Keep chunking/embedding/retrieval comparisons as supporting research or future work, not the main product identity.

## Trade-Offs And Challenges

- **Accuracy vs response time:** more context can improve answers but may slow API calls.
- **Chunk size vs retrieval quality:** smaller chunks may retrieve precise evidence; larger chunks preserve more context.
- **Local models vs API models:** local models improve privacy but require more compute; APIs improve quality and speed but require keys.
- **Simple MVP vs advanced platform:** the current project favors student workflow clarity over many research controls.
- **One PDF vs multiple PDFs:** one PDF keeps UX and persistence simple; multiple PDFs would require document management.
- **English-first vs Hebrew/RTL:** English PDFs are currently safer; Hebrew/RTL support needs dedicated testing.
- **UI simplicity vs many controls:** too many controls can distract students; core study actions should stay prominent.

## Proposed Solution

The proposed system uses a layered architecture:

1. PDF processing layer: upload, temporary file handling, PyMuPDF/pypdf extraction.
2. Study planning layer: convert pages into study sessions with summaries, objectives, concepts, and timing.
3. AI Tutor/RAG layer: provide general tutoring and section-grounded answers using PDF context where available.
4. Quiz/exam generation layer: generate practice questions and final exams with fallback behavior.
5. Progress tracking layer: record completed sessions, quiz scores, timer, final exam result, weak topics.
6. Streamlit UI layer: present the student workflow through six clear pages.
7. Local persistence layer: save and restore study progress in JSON.

Data flow:

`PDF -> text extraction -> study plan -> study sessions -> AI Tutor/quizzes -> final exam -> progress dashboard`

## Development Stages

1. **Clean repository and product identity**
   - What: separate MVP from legacy research code, update README and navigation.
   - Dependencies: current codebase.
   - Output: clean student-facing app structure.

2. **Improve PDF extraction and study plan generation**
   - What: robust text extraction, fallback handling, session suggestion, manual session count.
   - Dependencies: PDF service and study service.
   - Output: configurable study plan.

3. **Improve Study Mode and per-section state**
   - What: independent explanation, quiz, and answer state per session.
   - Dependencies: section state service.
   - Output: stable section-by-section workflow.

4. **Improve AI Tutor with PDF/section context**
   - What: separate general tutor and current-section modes.
   - Dependencies: GeneralAIService and study context extraction.
   - Output: more relevant tutoring answers.

5. **Add interactive final exam grading**
   - What: answer form, scoring, weak topic/section detection.
   - Dependencies: ExamService and ExamGradingService.
   - Output: actionable exam review.

6. **Add progress persistence**
   - What: local JSON save/restore for plan and progress.
   - Dependencies: ProgressService, PersistenceService.
   - Output: resumable study sessions.

7. **Polish UI/UX**
   - What: colors, cards, states, clearer empty/success messages.
   - Dependencies: Streamlit UI modules.
   - Output: polished MVP demo.

8. **Testing, evaluation, documentation, final presentation**
   - What: service tests, manual flow testing, proposal/docs.
   - Dependencies: stable MVP.
   - Output: tested and documented project.

## Testing And Evaluation Plan

- Unit tests for study, exam, quiz, progress, persistence, and navigation services.
- PDF workflow tests for extraction fallback and empty text.
- Session count suggestion tests.
- Quiz grading tests.
- Final exam grading tests.
- Progress persistence save/restore tests.
- Manual user flow testing across all pages.

Evaluation metrics:

- Answer relevance.
- Grounding to PDF/section context.
- Quiz correctness.
- Final exam grading correctness.
- Response time.
- User progress and learning gain.
