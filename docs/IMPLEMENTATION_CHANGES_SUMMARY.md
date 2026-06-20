# Smart Study Assistant - Implementation Changes Summary

This document summarizes the major implementation changes added to the Smart Study Assistant project during the recent development phase. It is intended to support the final project report, code review, and demonstration by explaining what changed, which files were affected, and what behavior the user should expect.

## 1. Summary of Modified Areas

| Area | Main files | What changed |
| --- | --- | --- |
| SQLite login and saved sessions | `services/database_service.py`, `services/auth_service.py`, `ui/state.py`, `ui/workflow.py`, `pages/auth_page.py` | Added local user accounts, per-user saved study sessions, persisted PDF bytes, saved progress, quizzes, exams, and restored sessions. |
| User isolation | `ui/state.py`, `services/database_service.py` | Prevented one signed-in user from seeing another user's active PDF/session data by resetting active study state when the authenticated user changes. |
| PDF restoration | `services/database_service.py`, `ui/state.py`, `ui/workflow.py` | Stored uploaded PDF bytes in SQLite so restored sessions can display page images and support section PDF downloads. |
| Runtime persistence | `services/database_service.py`, `ui/state.py`, `ui/workflow.py` | Persisted `current_section_index`, timer/progress state, completed sections, quiz scores, final exam state, and section outputs across refreshes. |
| Context-aware AI Tutor | `services/context_retrieval_service.py`, `ui/workflow.py`, `pages/ai_tutor_page.py` | Reworked PDF question answering so prompts use the active uploaded PDF/session context instead of unrelated or hardcoded content. |
| Smart intent retrieval | `services/context_retrieval_service.py`, `ui/workflow.py` | Added intent detection for chapter summaries, study section summaries, study-plan questions, document overview questions, and normal factual questions. |
| Grounded prompting | `ui/workflow.py`, `services/general_ai_service.py`, `docs/AI_PROMPTING_AND_DATA_FLOW.md` | Improved prompts so PDF answers are constrained to retrieved PDF chunks or saved study-section metadata. |
| Study mode UI | `pages/study_mode_page.py` | Made the rendered PDF/slides preview independently scrollable inside the left study column. |
| Documentation | `README.md`, `docs/AI_PROMPTING_AND_DATA_FLOW.md`, this file | Added technical documentation for prompting, data flow, saved sessions, retrieval behavior, and recent implementation changes. |
| Tests | `tests/test_mvp_recovery.py`, `tests/test_navigation.py`, related test files | Added and updated tests for PDF extraction/rendering recovery, saved-session behavior, provider selection, and navigation contract. |

## 2. Saved Sessions and Database Behavior

The project now uses a local SQLite database file named `.smartstudy.db` for authenticated users. The database is managed by `DatabaseService`.

### Data saved locally

| Data | Saved? | Location |
| --- | --- | --- |
| User accounts | Yes | `users` table |
| Passwords | Hashed only | `users.password_hash` |
| Uploaded PDF bytes | Yes | `documents.pdf_bytes` |
| PDF filename | Yes | `documents.filename` |
| Extracted section text | Yes | `study_sections.section_text` |
| Study section metadata | Yes | `study_sections` table |
| Current section index | Yes | `study_sessions.current_section_index` |
| Timer and progress state | Yes | `study_sessions.progress_state` and `section_progress` |
| Quiz attempts and scores | Yes | `quiz_attempts` and `section_progress` |
| Final exam attempts and results | Yes | `exam_attempts` |

The previous JSON persistence file remains as a fallback or legacy path for anonymous/non-authenticated usage, but logged-in users are restored through SQLite.

### Important fixes

- Different logged-in users no longer share the same active PDF state in the browser session.
- Uploaded PDFs are saved as SQLite BLOBs, so restored sessions can render page images instead of showing only extracted text.
- Refreshing the app now restores the active study section and accumulated study time.
- The latest SQLite session is automatically restored for a logged-in user when no active PDF/session is already loaded.

## 3. AI Tutor and PDF Question Answering

The AI Tutor was modified to answer from the active uploaded PDF and saved study sections.

### Retrieval flow

1. The user uploads a PDF.
2. `PdfService` extracts text from pages.
3. `StudyService` creates study sections with page ranges and metadata.
4. `ContextRetrievalService.build_chunks_from_pages()` builds local chunks from the active PDF pages.
5. `ContextRetrievalService.detect_query_intent()` classifies the question.
6. `ui/workflow.py` selects the appropriate answer path:
   - chapter summary
   - study section summary
   - study plan
   - document overview
   - normal factual PDF question
7. Only selected PDF chunks or saved section metadata are inserted into the AI prompt.
8. `GeneralAIService` calls the configured provider when available.
9. If the provider fails, local fallback answers are generated from retrieved context when possible.
10. Sources are appended to the answer so the user can see which pages or sections were used.

### Grounding rules

PDF-grounded prompts instruct the model to:

- answer only from provided PDF context;
- avoid outside knowledge;
- avoid guessing;
- return the standard not-enough-information message when the retrieved context does not support an answer;
- include source information based on retrieved pages or sections.

The standard unsupported-answer message is:

```text
The uploaded PDF does not contain enough information to answer this question.
```

## 4. Intent-Based Retrieval Behavior

The retrieval service now handles more than simple keyword overlap.

| User request type | Example | Implementation behavior |
| --- | --- | --- |
| Factual PDF question | "What is dynamic programming?" | Retrieves top matching chunks by text/title/key-concept overlap. |
| PDF overview | "What are the 5 main ideas in the PDF?" | If direct retrieval is weak, selects overview chunks across sections. |
| Chapter summary | "What is the main idea of chapter 4?" | Looks for chapter headings; if none exist, maps chapter number to study section number as a local fallback. |
| Study section summary | "Summarize section 3" | Uses saved study section metadata and section text. |
| Study plan request | "Help me prepare for this PDF" | Uses saved study sections, estimated time, difficulty, and key concepts. |
| General non-PDF question | "Explain recursion with an example" | Can be handled as general tutor mode when the user is not asking from the uploaded PDF. |

This fixes the previous issue where broad questions such as "what are the 5 main ideas in the PDF?" could fail even when extracted PDF text existed.

## 5. Prompting and AI Provider Changes

The project supports OpenAI first, then Groq, then deterministic/local fallback where implemented.

Provider selection is handled in `services/general_ai_service.py`. Prompt construction for section explanations, PDF answers, chapter summaries, section summaries, and study-plan responses is handled mainly in `ui/workflow.py`.

The detailed prompting documentation is maintained separately in:

```text
docs/AI_PROMPTING_AND_DATA_FLOW.md
```

That file explains what user input, PDF text, section metadata, conversation history, and output constraints are sent to the AI model.

## 6. UI Changes

The Study Mode PDF preview now scrolls independently.

Implementation:

```python
PDF_PREVIEW_SCROLL_HEIGHT = 720

with st.container(height=PDF_PREVIEW_SCROLL_HEIGHT, border=True):
    for offset, image in enumerate(images, start=section.start_page):
        st.image(image, caption=source_label(section, offset), width="stretch")
```

This keeps rendered PDF pages/slides inside the left preview area while the right-side study controls remain easier to access.

## 7. Error Handling and Fallbacks

| Failure case | Current behavior |
| --- | --- |
| PDF page rendering fails | Shows `Page images are unavailable. Use the extracted text fallback below.` and displays extracted text. |
| PDF section export fails | Shows a section PDF unavailable message instead of crashing. |
| AI provider missing | Returns a setup/fallback message from `GeneralAIService` or uses local fallback where available. |
| AI provider call fails | Uses local retrieved-context summaries where implemented. |
| Retrieved context is empty | Returns the not-enough-information message. |
| Final exam AI JSON is malformed | Falls back to deterministic locally generated exam content. |
| Refresh/browser rerun | Restores saved SQLite session state for logged-in users when available. |

## 8. Verification Performed

Recent verification included:

```bash
python3 -m py_compile pages/study_mode_page.py
python3 -m unittest tests.test_navigation tests.test_mvp_recovery
python3 -m unittest discover -s tests
```

The targeted UI and recovery tests passed after the scrollable PDF preview change. Earlier full test runs also passed after the runtime-state persistence change.

## 9. Remaining Limitations

| Limitation | Current status | Suggested improvement |
| --- | --- | --- |
| SQLite database is local only | Implemented for demo/local use | Add cloud storage or deployment-specific database configuration. |
| PDF bytes are stored locally | Implemented | Add a user-facing delete/export option and document retention policy. |
| Retrieval is lexical, not embedding-based | Implemented intentionally for simplicity | Add optional vector retrieval or hybrid retrieval for better semantic matching. |
| Prompt schemas are not centralized | Not currently implemented | Move prompts into a prompt manager with version numbers. |
| Prompt logging/debug mode | Not currently implemented | Add opt-in debug logs for prompt input, model output, and final UI answer. |
| Strict response schema validation for all AI calls | Partially implemented | Add JSON schema validation where structured AI output is required. |
| End-to-end browser testing | Not currently implemented | Add Playwright or Streamlit app tests for upload, restore, tutor, and exam flows. |

## 10. Summary

The recent implementation work moved Smart Study Assistant closer to a complete AI-based study product. The app now supports user-specific saved sessions, restores uploaded PDFs and progress after refresh, routes AI Tutor questions through the active PDF context, handles broad PDF-summary questions more reliably, and improves the Study Mode interface with an independently scrollable PDF/slides preview.

Documenting these changes is important because the project relies on external AI models, local PDF processing, and persisted user data. Clear implementation documentation improves reliability, privacy review, debugging, reproducibility, and academic evaluation.
