# Smart Study Assistant

Smart Study Assistant is a student-facing Streamlit MVP that turns an uploaded PDF into a guided study workflow.

Main workflow:

`Upload PDF -> Study Plan -> Study Mode -> AI Tutor -> Final Exam -> Dashboard`

The project is now organized around the student experience instead of exposing RAG benchmark controls in the main app.

## Current MVP Features

- Upload a PDF and extract text with PyMuPDF plus pypdf fallback.
- Generate a readable study plan with a suggested number of study sessions based on PDF size.
- Adjust the number of sessions before creating the final study plan.
- Study one section at a time while viewing rendered PDF pages and extracted text.
- Download the current section as a smaller PDF.
- Generate explanations and quizzes for each section.
- Ask general study questions in the AI Tutor.
- Generate a final exam from the study material with safe fallback behavior when AI output is unavailable or malformed.
- Track completed sections, quiz averages, study time, weak-topic recommendations, and final exam score.

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
  general_ai_service.py AI Tutor provider selection
  quiz_service.py       Deterministic section quiz generation

core/
  models.py             Shared document models

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

The AI Tutor and Final Exam generation choose providers in this order:

1. `OPENAI_API_KEY`
2. `GROQ_API_KEY`
3. Clear fallback/setup message when no key exists

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
