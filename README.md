# Smart Study Assistant

Smart Study Assistant is a student-facing Streamlit MVP that turns an uploaded PDF into a guided study workflow.

Main workflow:

`Upload PDF -> Study Plan -> Study Mode -> AI Tutor -> Final Exam -> Dashboard`

The project is now organized around the student experience instead of exposing RAG benchmark controls in the main app.

## Product Identity

Smart Study Assistant is an AI-powered PDF study assistant for students. The main product is not a RAG benchmark dashboard; RAG and document-grounded AI are technical foundations used behind the scenes to help students study from uploaded course material.

The current MVP focuses on a complete learning loop: upload a PDF, generate configurable study sessions, study each session, ask the AI Tutor, practice with quizzes, take an interactive final exam, and review progress in the dashboard.

Updated proposal documents:

- [Project proposal outline](docs/project_proposal_outline.md)
- [Hebrew project proposal](docs/project_proposal_he.md)
- [AI prompting and data flow documentation](docs/AI_PROMPTING_AND_DATA_FLOW.md)
- [Implementation changes summary](docs/IMPLEMENTATION_CHANGES_SUMMARY.md)

## Current MVP Features

- Switch the full application between English and Hebrew from the top navigation.
- Apply RTL layout automatically when Hebrew is selected.
- Upload a PDF and extract text with PyMuPDF plus pypdf fallback.
- Generate a readable study plan with a suggested number of study sessions based on PDF size.
- Adjust the number of sessions before creating the final study plan.
- Study one section at a time while viewing rendered PDF pages and extracted text.
- Download the current section as a smaller PDF.
- Generate explanations and quizzes for each section.
- Ask grounded PDF questions that retrieve only the most relevant local chunks before calling AI.
- Generate a final exam from the study material with safe fallback behavior when AI output is unavailable or malformed.
- Track completed sections, quiz averages, study time, weak-topic recommendations, and final exam score.

## Bilingual Support

The app supports English and Hebrew throughout the student workflow. Use the language selector in the top navigation to switch at any time:

- `English 🇺🇸`
- `עברית 🇮🇱`

The selected language is stored in Streamlit session state as `st.session_state.language`. UI labels, navigation, status messages, dashboard text, study plans, tutor prompts, quiz generation, final exam generation, and fallback messages use the active language.

Hebrew mode applies right-to-left layout and right-aligned text dynamically. New generated study plans, section quizzes, AI Tutor responses, and final exams are instructed to use the selected language.

## Context-Aware PDF Retrieval

After PDF extraction, the full extracted text is stored locally in Streamlit session state and local progress storage. The AI does not receive the whole PDF for every question. For PDF-grounded questions, the app searches the uploaded PDF locally with `ContextRetrievalService`, scores section/page chunks by word overlap, section title matches, and key concept matches, then sends only the top relevant chunks to the AI.

If no relevant PDF chunks are found, the app returns: `The uploaded PDF does not contain enough information to answer this question.` This reduces hallucinations, API cost, privacy risk, and irrelevant context. "Explain This Section" intentionally keeps its previous behavior and sends only the current section title, page range, key concepts, and section text.

## Smart Intent-Based Retrieval

AI Tutor and current-section questions now detect the user's intent before using normal chunk retrieval. Factual questions still use strict local chunk retrieval and return the not-enough-information message when the PDF does not support an answer.

Chapter and section summary requests use document structure detection instead of relying only on word overlap. For example, "What is the main idea of chapter 4?" searches for real headings such as `Chapter 4`, `CHAPTER 4`, `Ch. 4`, `Chapter Four`, or numbered headings near the start of PDF pages. If real chapter headings are not detected, the app maps `chapter 4` to Study Section 4 as the closest local fallback and marks the source accordingly.

Study-plan requests use the saved study sections, including titles, summaries, key concepts, difficulty, and estimated time, instead of random retrieved chunks. The assistant remains grounded: it answers only from extracted PDF text or saved section metadata and does not use outside knowledge.

Fixed behavior: before this change, questions like "What is the main idea of chapter 4?" could return `The uploaded PDF does not contain enough information to answer this question.` even when Study Section 4 existed. Now the app detects the chapter request, retrieves chapter 4 or Study Section 4, summarizes it, and displays the source pages.

## SQLite Login And Saved Sessions

The app now supports simple local accounts backed by SQLite. Users register or log in before using study features, and each user's uploaded documents and study sessions are kept separate in `.smartstudy.db`.

Saved data includes the uploaded PDF bytes, generated study sections, extracted section text, section completion, explanation text, quiz attempts, quiz scores, final exam attempts, final exam answers, and weak-topic results. Storing the PDF bytes locally lets saved sessions restore page images and section PDF downloads after the user clicks Continue. Passwords are hashed before storage; plain-text passwords are never written to the database.

SQLite is local and lightweight, which fits the academic/demo scope of this project. The existing JSON persistence remains as a fallback/legacy path, while logged-in users use SQLite as the preferred persistence layer.

## Improved Study Sectioning And Time Estimation

Study sections are validated for page coverage, ordering, non-overlapping page ranges, useful titles, summaries, key concepts, and learning objectives. If AI sectioning fails validation, the app falls back to deterministic heuristic sectioning.

Estimated study time is calculated from section word count, difficulty, key concept count, and practice/review time. This makes estimates consistent and explainable instead of relying blindly on AI-provided values.

## Project Architecture

```text
ui/
  streamlit_app.py      Streamlit entry point and page router
  navigation.py         Navigation contract
  styles.py             Shared CSS
  state.py              Session state helpers
  components.py         Shared Streamlit UI components
  workflow.py           UI workflow helpers that call services

pages/
  upload_page.py
  study_plan_page.py
  study_mode_page.py
  ai_tutor_page.py
  final_exam_page.py
  dashboard_page.py

services/
  study_service.py      Study plan generation
  exam_service.py       Final exam generation and fallback handling
  progress_service.py   Progress, timers, quiz averages
  pdf_service.py        PDF text extraction
  pdf_render_service.py PDF page rendering
  pdf_section_service.py Section PDF extraction
  context_retrieval_service.py Local PDF chunk retrieval for grounded AI prompts
  database_service.py   SQLite users, sessions, progress, quizzes, and exams
  auth_service.py       Login, registration, logout, and password hashing
  general_ai_service.py AI Tutor provider selection
  quiz_service.py       Deterministic section quiz generation

core/
  models.py             Shared document models

translations.py         English/Hebrew translation table and language helpers

research/
  Legacy RAG, LangChain, benchmark, vector store, embedding, and experiment code.
```

## Run The App

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the student assistant:

```bash
streamlit run ui/streamlit_app.py
```

## Optional AI Keys

AI study-plan sectioning, the AI Tutor, quiz generation, and Final Exam generation choose providers in this order:

1. `OPENAI_API_KEY`
2. `GROQ_API_KEY`
3. Deterministic offline fallback or clear setup message when no key exists

Unit tests do not make real API calls.

## Verification

Run the MVP tests:

```bash
python -m unittest discover -s tests
```

Compile the main app and services:

```bash
python -m py_compile ui/*.py pages/*.py services/*.py core/*.py
```

## What Changed From The Older RAG Platform

- The main app no longer presents RAG Check, OCR, benchmark Results, or experimentation controls as student-facing pages.
- Legacy RAG and LangChain experimentation code was moved into `research/`.
- The Streamlit app was split into page modules and shared helpers instead of one large file.
- The top-level service layer now emphasizes the student workflow: PDF extraction, study planning, AI tutoring, exams, quizzes, and progress.
- Tests now focus on MVP behavior instead of internal Streamlit implementation strings.

## Future Work

- Add persistent user sessions so progress survives browser restarts.
- Improve scanned-PDF support with a dedicated OCR preprocessing flow.
- Add richer exam grading and review analytics.
- Add export options for study plans, quiz results, and final exam reports.
- Add lightweight end-to-end UI checks for the Streamlit workflow.
