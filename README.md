# Smart Study Assistant

Smart Study Assistant is an AI-powered personalized study platform for PDF
course material. It transforms static notes, slides, textbook chapters, and
papers into an adaptive learning workflow with a study plan, guided section
study, grounded explanations, quizzes, final exam practice, citations, and
local progress tracking.

The `Ask AI` page is a general AI tutor powered by OpenAI when
`OPENAI_API_KEY` is available, with Groq as a fallback. Study Mode, quizzes,
and final exam practice remain grounded in the uploaded PDF.

## Project Goal

Help students study uploaded course material step by step:

1. Upload a PDF.
2. Generate a logical study plan.
3. Study one section at a time.
4. Choose the explanation level that fits the goal.
5. Test understanding with mini quizzes.
6. Explain each section in your own words before moving on.
7. Review mistakes and weak topics.
8. Track study time, progress, and exam readiness.
9. Finish with a PDF-grounded practice exam.

## Main Features

- PDF upload with normal text extraction and optional OCR.
- Local chunking, embeddings, vector search, and grounded retrieval.
- Automatic study plan with section title, page range, summary, key concepts,
  estimated study time, and difficulty.
- Course-roadmap Study Plan with section cards, status badges, progress, and
  key-concept chips.
- Guided section-by-section study mode.
- Split-screen Study Mode with the original PDF visible beside the AI study
  assistant.
- One-click section explanations with key definitions, important points,
  example questions, and source previews.
- Study session timer with start, pause, resume, finish, and actual time
  tracking per section.
- Understanding checks that score the student's own explanation against the
  current PDF section.
- Exam Focus boxes with important points, possible questions, common mistakes,
  and key terms to memorize.
- Section mini quizzes with multiple choice, true/false, and short-answer
  questions.
- General `Ask AI` tutor for examples, explanations, and open-ended study help.
- Automatic quiz scoring with correct answers, explanations, review topics, and
  mistake-based review lessons.
- Final exam mode with easy, medium, and hard questions across the full PDF.
- Open-question grading by PDF-grounded keyword overlap and feedback.
- Markdown study-pack export from the Dashboard.
- Local progress tracking in `.cache/progress.json`.
- Dashboard with progress, completed sections, average quiz score, weak topics,
  strong topics, repeated mistakes, recommended review sections, timing stats,
  recommended next section, and exam readiness.
- Clean source citations such as `Source: Page 5`.

## Technologies Used

- Python
- Streamlit
- Custom Streamlit CSS for a calm navy/teal education dashboard design
- PyMuPDF and pypdf for PDF text extraction
- pytesseract and optional EasyOCR for OCR
- SentenceTransformers MiniLM embeddings, with mock embedding fallback
- FAISS or in-memory vector store
- OpenAI or Groq API for the general AI tutor
- Groq OpenAI-compatible API for optional JSON exam generation
- Python `unittest` test suite

## How To Run

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python -m streamlit run ui/streamlit_app.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Configuration

The app reads configuration from environment variables where possible.

Important defaults:

```text
GROQ_MODEL=llama-3.1-8b-instant
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=minilm
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_BACKEND=faiss
CHUNK_SIZE=700
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=4
MIN_RETRIEVAL_SCORE=0.08
LLM_TEMPERATURE=0.2
GENERAL_AI_TEMPERATURE=0.4
LLM_MAX_TOKENS=2200
```

Ask AI uses a general AI API. OpenAI is preferred when available:

```bash
export OPENAI_API_KEY=your-openai-api-key
export OPENAI_MODEL=gpt-4o-mini
```

Groq is used as a fallback for Ask AI and is also used for AI final exam
generation when available:

```bash
export GROQ_API_KEY=your-groq-api-key
export GROQ_MODEL=llama-3.1-8b-instant
```

You can also copy the example file:

```bash
cp config/groq_api_key_example.txt config/groq_api_key.txt
```

Then put your real key in `config/groq_api_key.txt`. The real key file is
ignored by Git.

## How The RAG Pipeline Works

```text
PDF upload
  -> page text extraction
  -> optional OCR for scanned or low-text pages
  -> overlapping page chunks
  -> local embeddings
  -> FAISS or memory vector store
  -> top-k retrieval
  -> relevance filtering
  -> grounded extractive answer with citations
```

Each chunk keeps:

- `chunk_id`
- `page_number`
- `source_id`
- character offsets where available
- extraction metadata
- chunking metadata
- study metadata after plan generation: `section_id` and `section_title`

PDF-grounded study workflows do not use outside knowledge. If retrieved
evidence is weak or missing, the app returns:

```text
I could not find this clearly in the uploaded material.
```

The separate `Ask AI` tab is intentionally general/open-ended and does not show
PDF citations unless you ask about the uploaded material. For PDF-grounded
answers, use Study Mode.

## How Study Sections Are Created

`StudyService.create_study_plan()` groups indexed chunks into logical study
sections using page order, chunk metadata, and extracted content signals. For
each section it creates:

- title
- page range
- short summary
- key concepts
- estimated study time
- difficulty
- source chunk IDs
- source preview

The Study Plan page presents these sections as a course roadmap. Each card shows
the section number, title, page range, estimated time, difficulty badge, short
summary, key-concept chips, progress status, and a button to start studying.

The service also writes the section ID and title into chunk metadata. That lets
section-specific study mode and section questions retrieve only chunks from the
active section.

## How Explanations Are Generated

Section explanations are grounded in the section's PDF chunks. Study Mode uses
one clear, student-friendly, exam-oriented explanation style with definitions,
important points, example questions, and source references.

In Study Mode, the current PDF section is cropped to the section page range and
rendered as page images on the left side of the page. The cropped section PDF
can also be downloaded. Extracted text remains available only as a
fallback/source preview.

## Study Timer And Understanding Checks

Each section has a local study timer:

- start timer
- pause timer
- finish section

The app stores actual time spent per section and compares it with the estimated
study time from the generated study plan. The Dashboard shows total study time,
average time per section, estimated vs actual time, and sections that took
longer than expected.

## How Quizzes Are Generated

Mini quizzes are generated from the active section only. They include:

- multiple choice
- true/false
- short answer
- correct answers
- answer explanations
- source references

When submitted, `StudyService.grade_quiz()` calculates the score percentage,
marks correct and wrong answers, and returns weak and strong topics.

The app stores wrong answers in progress, identifies weak topics, and recommends
review sections in the Dashboard.

## How Final Exams Are Generated

Final Exam mode uses Groq AI when an API key is available. If the AI call fails,
the app falls back to deterministic PDF-grounded questions so the demo remains
stable.

The local final exam includes easy, medium, and hard questions with multiple
choice, short answer, and open explanation prompts. Open answers are evaluated
by overlap with expected PDF-grounded concepts and receive feedback.

The Dashboard calculates exam readiness using section completion, average quiz
score, understanding check scores, final exam score when available, and a weak
topic penalty. It displays a readiness percentage, a status label (`Not ready`,
`Needs review`, `Almost ready`, or `Ready`), and a recommended next action.

## Progress Tracking

Progress is stored locally in:

```text
.cache/progress.json
```

Tracked fields include:

- uploaded document name
- number of sections
- completed sections
- quiz scores per section
- final exam score
- weak topics
- strong topics
- last studied section
- actual study time per section
- understanding check scores
- mistake history
- recommended review sections
- total progress percentage

## Export Study Pack

The Dashboard includes an `Export Study Pack` button. It downloads a Markdown
file containing:

- study plan
- section summaries
- key concepts and definitions where available
- exam focus points
- flashcards
- quiz scores
- weak topics and review plan
- recent mistake history

## Demo Workflow

The `data/` folder includes sample PDFs. A simple demo flow:

1. Open the app.
2. Go to `Upload PDF`.
3. Upload `data/example.pdf` or `data/40-algorithms.pdf`.
4. Click `Process PDF and Generate Study Plan`.
5. Open `Study Plan` and review generated sections.
6. Open `Study Mode`, start the first section, and generate an explanation.
7. Start the timer, study while viewing the PDF, then finish the section.
8. Click `Generate Quiz`, answer it, and submit.
9. Open `Ask AI` and ask a general study question.
10. Open `Dashboard` to see progress, timing, weak topics, and readiness.
11. Export a Markdown study pack.
12. Open `Final Exam` and generate a full practice exam.

## Current Limitations

- Study sectioning is heuristic and may not perfectly match human chapter
  boundaries for every PDF.
- Local quiz and exam generation is grounded and useful, but less flexible than
  a hosted LLM.
- OCR quality depends on the input scan and installed OCR tools.
- Progress is local to this project workspace and not multi-user.
- The vector index is rebuilt after app restart.
- Study Mode renders PDF sections as images from cropped PDF bytes. If image
  rendering fails, the extracted text fallback remains available.
- Understanding checks and open-question grading use grounded keyword overlap,
  not a full human-level evaluator.

## Future Improvements

- Persist uploaded documents and vector indexes across restarts.
- Add browser `localStorage` synchronization for progress.
- Add richer LLM-generated explanations with strict citation validation.
- Support instructor-defined learning objectives.
- Add spaced repetition and due dates.
- Add optional PDF export for study plans and exam reports.
- Add multi-document course collections with per-course dashboards.
