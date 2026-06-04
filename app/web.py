from __future__ import annotations

import base64
import cgi
import json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from services.exam_service import ExamGenerationError, ExamRequest, FullExamService
from services.rag_service import PDFIndex, PDFRAGService, RAGPipelineError
from services.pdf_service import OCRMode, PdfService

logging.basicConfig(level=logging.WARNING, format="%(message)s")


@dataclass
class AppState:
    index: PDFIndex | None = None


class SmartStudyAPI:
    def __init__(self) -> None:
        self._state = AppState()
        self._lock = threading.Lock()
        self._rag = PDFRAGService()
        self._exam = FullExamService()

    def upload_pdf(self, filename: str, pdf_bytes: bytes, ocr_mode: OCRMode = "auto") -> dict:
        with self._lock:
            self._state.index = self._rag.build_index_from_upload(pdf_bytes, filename, ocr_mode=ocr_mode)
            return {"ok": True, "index": self._state.index.to_summary()}

    def upload_pdfs(self, uploads: list[tuple[str, bytes]], ocr_mode: OCRMode = "auto") -> dict:
        with self._lock:
            self._state.index = self._rag.build_index_from_uploads(uploads, ocr_mode=ocr_mode)
            return {"ok": True, "index": self._state.index.to_summary()}

    def ask(self, question: str) -> dict:
        index = self._require_index()
        return self._rag.answer(index, question).to_dict()

    def generate_exam(self, payload: dict) -> dict:
        index = self._require_index()
        question_types = payload.get("question_types", [])
        if isinstance(question_types, str):
            question_types = [question_types]
        request = ExamRequest(
            number_of_questions=int(payload.get("number_of_questions", payload.get("question_count", 10))),
            question_types=list(question_types),
            difficulty=str(payload.get("difficulty", "mixed")),
            include_answer_key=self._bool_payload(payload.get("include_answer_key", True)),
            multiple_choice=int(payload["multiple_choice"]) if "multiple_choice" in payload else None,
            open_questions=int(payload["open_questions"]) if "open_questions" in payload else None,
            true_false=int(payload["true_false"]) if "true_false" in payload else None,
            short_answer=int(payload["short_answer"]) if "short_answer" in payload else None,
        )
        return {"ok": True, "exam": self._exam.generate_exam(index, request)}

    def sources(self) -> dict:
        index = self._require_index()
        return {
            "pdf_name": index.pdf_name,
            "chunks": [
                {
                    "pdf_name": chunk.source_id or index.pdf_name,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
                for chunk in index.chunks
            ],
        }

    def status(self) -> dict:
        if not self._state.index:
            return {"ok": True, "indexed": False}
        return {"ok": True, "indexed": True, "index": self._state.index.to_summary()}

    def _require_index(self) -> PDFIndex:
        if not self._state.index:
            raise RAGPipelineError("Upload and process a PDF before using this endpoint.")
        return self._state.index

    @staticmethod
    def _bool_payload(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)


APP = SmartStudyAPI()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"ok": True})
            return
        if path == "/api/status":
            self._send_json(APP.status())
            return
        if path == "/api/sources":
            self._handle_json_call(APP.sources)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload":
            self._handle_upload()
            return
        if path == "/api/ask":
            payload = self._read_json()
            self._handle_json_call(lambda: APP.ask(str(payload.get("question", ""))))
            return
        if path == "/api/exam":
            payload = self._read_json()
            self._handle_json_call(lambda: APP.generate_exam(payload))
            return
        if path in {"/api/generate-exam", "/api/generate-quiz", "/generate-exam", "/generate-quiz"}:
            payload = self._read_json()
            self._handle_json_call(lambda: APP.generate_exam(payload))
            return
        self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        return

    def _handle_upload(self) -> None:
        try:
            uploads, ocr_mode = self._read_upload_request()
            self._send_json(APP.upload_pdfs(uploads, ocr_mode=ocr_mode))
        except (RAGPipelineError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_json_call(self, callback) -> None:
        try:
            self._send_json(callback())
        except (RAGPipelineError, ExamGenerationError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _read_upload_request(self) -> tuple[list[tuple[str, bytes]], OCRMode]:
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            file_items = form["file"] if "file" in form else None
            if file_items is None:
                raise ValueError("Upload field 'file' is required.")
            if not isinstance(file_items, list):
                file_items = [file_items]
            uploads = [
                (file_item.filename, file_item.file.read())
                for file_item in file_items
                if getattr(file_item, "filename", "")
            ]
            if not uploads:
                raise ValueError("Upload field 'file' is required.")
            ocr_mode = self._normalize_ocr_mode(form.getfirst("ocr_mode", "auto"))
            return uploads, ocr_mode

        payload = self._read_json()
        ocr_mode = self._normalize_ocr_mode(payload.get("ocr_mode", "auto"))
        files = payload.get("files") or payload.get("pdfs")
        if isinstance(files, list):
            uploads = []
            for item in files:
                if not isinstance(item, dict):
                    raise ValueError("Each file entry must be an object.")
                filename = str(item.get("filename", "uploaded.pdf"))
                encoded = str(item.get("pdf_base64", ""))
                if not encoded:
                    raise ValueError("Each file entry requires pdf_base64.")
                uploads.append((filename, base64.b64decode(encoded)))
            return uploads, ocr_mode

        filename = str(payload.get("filename", "uploaded.pdf"))
        encoded = str(payload.get("pdf_base64", ""))
        if not encoded:
            raise ValueError("Provide pdf_base64 or multipart field 'file'.")
        return [(filename, base64.b64decode(encoded))], ocr_mode

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw or b"{}")

    def _send_json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    @staticmethod
    def _normalize_ocr_mode(value) -> OCRMode:
        try:
            return PdfService._normalize_ocr_mode(str(value or "auto"))
        except Exception as exc:
            raise ValueError(str(exc)) from exc


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Smart Study Assistant API running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
