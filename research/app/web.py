from __future__ import annotations

import html
import json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.pdf_service import PdfService
from services.retrieval_service import RetrievalService
from services.vector_store_service import VectorStoreService

logging.basicConfig(level=logging.WARNING, format="%(message)s")


@dataclass
class AssistantIndex:
    pdf_name: str
    page_count: int
    chunk_count: int
    retrieval_service: RetrievalService


class SmartStudyApp:
    def __init__(self) -> None:
        self._indexes: dict[str, AssistantIndex] = {}
        self._lock = threading.Lock()

    @property
    def pdfs(self) -> list[Path]:
        return sorted(Path("data").glob("*.pdf"))

    def get_index(self, pdf_name: str) -> AssistantIndex:
        pdf_path = Path("data") / pdf_name
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Choose one of the PDFs in the data folder.")

        with self._lock:
            if pdf_name in self._indexes:
                return self._indexes[pdf_name]

            pdf_service = PdfService()
            chunk_service = ChunkService(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            embedding_service = EmbeddingService()
            vector_store = VectorStoreService()

            pages = pdf_service.extract_pages(pdf_path)
            chunks = chunk_service.chunk_pages(pages)
            embeddings = embedding_service.embed_texts(chunks)
            vector_store.add(chunks, embeddings)

            index = AssistantIndex(
                pdf_name=pdf_name,
                page_count=len(pages),
                chunk_count=len(chunks),
                retrieval_service=RetrievalService(
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                ),
            )
            self._indexes[pdf_name] = index
            return index

    def answer(self, pdf_name: str, question: str, top_k: int) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Ask a question first.")

        index = self.get_index(pdf_name)
        response = index.retrieval_service.retrieve(question, top_k=top_k)
        source_texts = [result.chunk.text for result in response.results]
        answer = "Based on the document:\n\n" + "\n\n".join(source_texts)[:900]

        return {
            "answer": answer,
            "pdf": index.pdf_name,
            "pages": index.page_count,
            "chunks": index.chunk_count,
            "sources": [
                {
                    "page": result.chunk.page_number,
                    "score": round(result.score, 3),
                    "text": result.chunk.text,
                }
                for result in response.results
            ],
        }


APP = SmartStudyApp()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(render_page(APP.pdfs))
            return

        if path == "/health":
            self._send_json({"ok": True})
            return

        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/ask":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = APP.answer(
                pdf_name=str(payload.get("pdf", "")),
                question=str(payload.get("question", "")),
                top_k=int(payload.get("top_k", 3)),
            )
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def render_page(pdfs: list[Path]) -> str:
    options = "\n".join(
        f'<option value="{html.escape(pdf.name)}">{html.escape(pdf.name)}</option>'
        for pdf in pdfs
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Study Assistant</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f2;
      --ink: #171812;
      --muted: #61665a;
      --panel: #ffffff;
      --line: #d9ddcf;
      --accent: #256d5a;
      --accent-strong: #174b3e;
      --warm: #e46f44;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }}

    .shell {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }}

    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: end;
      margin-bottom: 24px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 4rem);
      line-height: 1;
      letter-spacing: 0;
    }}

    .subhead {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 680px;
      font-size: 1rem;
      line-height: 1.5;
    }}

    .status {{
      color: var(--accent-strong);
      border: 1px solid var(--line);
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.62);
      white-space: nowrap;
      font-size: 0.92rem;
    }}

    main {{
      display: grid;
      grid-template-columns: minmax(0, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 14px 32px rgba(23, 24, 18, 0.07);
    }}

    label {{
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }}

    select, textarea, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      color: var(--ink);
      background: white;
    }}

    textarea {{
      min-height: 160px;
      resize: vertical;
      line-height: 1.45;
    }}

    .field {{ margin-bottom: 16px; }}

    .row {{
      display: grid;
      grid-template-columns: 1fr 96px;
      gap: 12px;
      align-items: end;
    }}

    button {{
      width: 100%;
      border: 0;
      border-radius: 8px;
      padding: 13px 14px;
      color: white;
      background: var(--accent);
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}

    button:hover {{ background: var(--accent-strong); }}
    button:disabled {{ opacity: 0.64; cursor: wait; }}

    .answer {{
      min-height: 260px;
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 1rem;
    }}

    .meta {{
      color: var(--muted);
      margin-bottom: 14px;
      font-size: 0.92rem;
    }}

    .sources {{
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }}

    .source {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}

    .source strong {{
      display: block;
      color: var(--warm);
      margin-bottom: 6px;
    }}

    .source p {{
      margin: 0;
      color: #30332b;
      line-height: 1.45;
    }}

    @media (max-width: 820px) {{
      header {{ align-items: start; flex-direction: column; }}
      main {{ grid-template-columns: 1fr; }}
      .status {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>Smart Study Assistant</h1>
        <p class="subhead">Ask questions against the PDFs in <code>data/</code>. This uses the project’s current mock embeddings, so treat the answer as a retrieval demo.</p>
      </div>
      <div class="status" id="status">Ready</div>
    </header>

    <main>
      <section class="panel">
        <div class="field">
          <label for="pdf">PDF</label>
          <select id="pdf">{options}</select>
        </div>

        <div class="field">
          <label for="question">Question</label>
          <textarea id="question">What is machine learning?</textarea>
        </div>

        <div class="row">
          <div class="field">
            <label for="topK">Sources</label>
            <input id="topK" type="number" min="1" max="8" value="3">
          </div>
          <button id="ask">Ask</button>
        </div>
      </section>

      <section class="panel">
        <div class="meta" id="meta">No answer yet.</div>
        <div class="answer" id="answer">Choose a PDF, ask a question, and the retrieved chunks will appear here.</div>
        <div class="sources" id="sources"></div>
      </section>
    </main>
  </div>

  <script>
    const askButton = document.getElementById("ask");
    const statusEl = document.getElementById("status");
    const metaEl = document.getElementById("meta");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");

    function escapeText(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\\"": "&quot;",
        "'": "&#039;"
      }}[char]));
    }}

    askButton.addEventListener("click", async () => {{
      askButton.disabled = true;
      statusEl.textContent = "Indexing and searching...";
      sourcesEl.innerHTML = "";

      try {{
        const response = await fetch("/api/ask", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            pdf: document.getElementById("pdf").value,
            question: document.getElementById("question").value,
            top_k: Number(document.getElementById("topK").value || 3)
          }})
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");

        metaEl.textContent = `${{data.pdf}} · ${{data.pages}} pages · ${{data.chunks}} chunks`;
        answerEl.textContent = data.answer;
        sourcesEl.innerHTML = data.sources.map((source, index) => `
          <article class="source">
            <strong>Source ${{index + 1}} · page ${{source.page}} · score ${{source.score}}</strong>
            <p>${{escapeText(source.text.slice(0, 520))}}</p>
          </article>
        `).join("");
        statusEl.textContent = "Ready";
      }} catch (error) {{
        metaEl.textContent = "Something went wrong.";
        answerEl.textContent = error.message;
        statusEl.textContent = "Error";
      }} finally {{
        askButton.disabled = false;
      }}
    }});
  </script>
</body>
</html>"""


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8501), RequestHandler)
    print("Smart Study Assistant front end: http://localhost:8501")
    server.serve_forever()


if __name__ == "__main__":
    main()
