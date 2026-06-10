from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag.prompts import GROUNDED_ANSWER_PROMPT


class LangChainPipelineError(Exception):
    """Raised when the LangChain RAG pipeline cannot complete a task."""


class LangChainDependencyError(LangChainPipelineError):
    """Raised when an optional LangChain dependency is unavailable."""


class _MockEmbeddings:
    """Deterministic local embeddings for offline demos and tests."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for index, char in enumerate((text or "")[:5000]):
            slot = (ord(char) + index) % self.dimension
            vector[slot] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class LangChainRAGPipeline:
    """Minimal LangChain-based PDF RAG pipeline for the Streamlit app."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "define",
        "did",
        "do",
        "does",
        "explain",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be at least 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.embedding_model_name = embedding_model_name.strip() or "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vectorstore: Any | None = None
        self.documents: list[Any] = []
        self.chunks: list[Any] = []
        self.stats: dict[str, Any] = {}
        self.prompt_template = GROUNDED_ANSWER_PROMPT

    def load_pdf(self, file_path: str) -> list[Any]:
        loader_class = self._get_pdf_loader_class()
        try:
            loader = loader_class(file_path)
            documents = loader.load()
        except Exception as exc:
            raise LangChainPipelineError(f"Could not load PDF: {exc}") from exc

        normalized: list[Any] = []
        for doc in documents:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata.setdefault("source", file_path)
            metadata["page"] = self._coerce_page_number(metadata.get("page"), zero_based=True)
            cleaned_text = self.clean_text(getattr(doc, "page_content", ""))
            if cleaned_text:
                normalized.append(self._create_document(cleaned_text, metadata))

        self.documents = normalized
        return normalized

    def clean_text(self, text: str) -> str:
        text = text or ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\|{2,}", "|", text)
        text = re.sub(r"[ \t]+", " ", text)

        cleaned_lines: list[str] = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip(" |")
            if not line:
                continue
            if len(re.sub(r"[^A-Za-z0-9]", "", line)) < 2:
                continue
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_documents(self, documents: list[Any]) -> list[Any]:
        splitter_class = self._get_text_splitter_class()
        splitter = splitter_class(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = splitter.split_documents(documents)

        normalized: list[Any] = []
        for index, doc in enumerate(split_docs, start=1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata.setdefault("chunk_id", f"chunk_{index}")
            cleaned_text = self.clean_text(getattr(doc, "page_content", ""))
            if cleaned_text:
                normalized.append(self._create_document(cleaned_text, metadata))

        self.chunks = normalized
        return normalized

    def build_vectorstore(self, chunks: list[Any]) -> Any:
        faiss_class = self._get_faiss_class()
        embeddings = self._build_embeddings()

        try:
            self.vectorstore = faiss_class.from_documents(chunks, embeddings)
        except Exception as exc:
            raise LangChainPipelineError(f"Could not build vector store: {exc}") from exc

        return self.vectorstore

    def process_pdf(self, file_path: str) -> dict[str, Any]:
        documents = self.load_pdf(file_path)
        if not documents:
            raise LangChainPipelineError("No text could be extracted from this PDF.")

        chunks = self.split_documents(documents)
        if not chunks:
            raise LangChainPipelineError("The PDF did not produce any retrievable chunks.")

        self.build_vectorstore(chunks)

        self.stats = {
            "pages": len(documents),
            "chunks": len(chunks),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        return self.stats

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        question = (question or "").strip()
        if not question:
            raise LangChainPipelineError("Question cannot be empty.")
        if self.vectorstore is None:
            raise LangChainPipelineError("Process a PDF before asking questions.")

        try:
            results = self.vectorstore.similarity_search_with_score(question, k=self.top_k)
        except AttributeError:
            results = [(doc, None) for doc in self.vectorstore.similarity_search(question, k=self.top_k)]
        except Exception as exc:
            raise LangChainPipelineError(f"Retrieval failed: {exc}") from exc

        formatted: list[dict[str, Any]] = []
        for doc, score in results:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            formatted.append(
                {
                    "text": self._trim_text(getattr(doc, "page_content", "").strip()),
                    "source": self._source_label(metadata.get("source")),
                    "page": self._coerce_page_number(metadata.get("page"), zero_based=False),
                    "score": float(score) if score is not None else None,
                }
            )
        return formatted

    def answer_question(self, question: str) -> dict[str, Any]:
        retrieved_chunks = self.retrieve(question)
        if not self._has_reliable_context(question, retrieved_chunks):
            topic = self._topic_label(question)
            answer = "I could not find a reliable answer in the uploaded document."
            if topic:
                answer = f"I could not find a reliable answer about {topic} in the uploaded document."
            return {
                "answer": answer,
                "citations": [],
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
            }

        answer = self._compose_answer(question, retrieved_chunks)
        citations = self._build_citations(retrieved_chunks)
        sources = self._build_sources(retrieved_chunks)
        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }

    def _build_embeddings(self) -> Any:
        if self._use_mock_embeddings:
            return _MockEmbeddings()

        embeddings_class = self._get_huggingface_embeddings_class()
        try:
            try:
                return embeddings_class(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except TypeError:
                return embeddings_class(model_name=self.embedding_model_name)
        except Exception as exc:
            raise LangChainPipelineError(
                "Could not load the HuggingFace embedding model. "
                "The first download can be slow. If this keeps failing, try mock mode."
            ) from exc

    def _create_document(self, page_content: str, metadata: dict[str, Any]) -> Any:
        document_class = self._get_document_class()
        return document_class(page_content=page_content, metadata=metadata)

    @property
    def _use_mock_embeddings(self) -> bool:
        normalized = self.embedding_model_name.strip().lower()
        return normalized in {"mock", "mock-embeddings", "offline-mock"}

    @staticmethod
    def _get_document_class() -> Any:
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise LangChainDependencyError(
                "LangChain document support is missing. Install the packages from requirements.txt."
            ) from exc
        return Document

    @staticmethod
    def _get_pdf_loader_class() -> Any:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as exc:
            raise LangChainDependencyError(
                "PyPDFLoader is unavailable. Install langchain-community and pypdf."
            ) from exc
        return PyPDFLoader

    @staticmethod
    def _get_text_splitter_class() -> Any:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise LangChainDependencyError(
                "RecursiveCharacterTextSplitter is unavailable. Install langchain-text-splitters."
            ) from exc
        return RecursiveCharacterTextSplitter

    @staticmethod
    def _get_faiss_class() -> Any:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise LangChainDependencyError(
                "FAISS vector store support is unavailable. Install langchain-community and faiss-cpu."
            ) from exc
        return FAISS

    @staticmethod
    def _get_huggingface_embeddings_class() -> Any:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise LangChainDependencyError(
                "HuggingFace embeddings support is unavailable. Install langchain-huggingface and sentence-transformers."
            ) from exc
        return HuggingFaceEmbeddings

    @classmethod
    def _extract_keywords(cls, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", (text or "").lower())
        return [token for token in tokens if token not in cls.STOPWORDS]

    @classmethod
    def _topic_label(cls, question: str) -> str:
        lowered = (question or "").strip().rstrip("? ")
        lowered = re.sub(
            r"^(what|why|how|when|where|who|define|explain)\s+(is|are|does|do|can)?\s*",
            "",
            lowered,
            flags=re.IGNORECASE,
        ).strip()
        lowered = re.sub(r"[^A-Za-z0-9 ]+", "", lowered).strip()
        if not lowered:
            return ""
        return lowered.upper() if len(lowered) <= 15 else lowered

    @classmethod
    def _has_reliable_context(cls, question: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
        if not retrieved_chunks:
            return False

        combined_text = "\n".join(chunk["text"] for chunk in retrieved_chunks).lower()
        keywords = cls._extract_keywords(question)
        if not keywords:
            return False

        topic_label = cls._topic_label(question).lower()
        if topic_label and len(topic_label.split()) > 1 and topic_label not in combined_text:
            return False

        matched_keywords = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", combined_text))
        coverage = matched_keywords / len(keywords)

        best_chunk_overlap = 0
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"].lower()
            overlap = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", chunk_text))
            best_chunk_overlap = max(best_chunk_overlap, overlap)

        return coverage >= 0.4 and best_chunk_overlap >= 1

    @classmethod
    def _compose_answer(cls, question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        keywords = cls._extract_keywords(question)
        sentence_candidates: list[tuple[int, str]] = []

        for chunk in retrieved_chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = sum(
                    1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", sentence.lower())
                )
                if overlap > 0:
                    sentence_candidates.append((overlap, sentence))

        sentence_candidates.sort(key=lambda item: (-item[0], len(item[1])))

        selected: list[str] = []
        seen = set()
        for _, sentence in sentence_candidates:
            if sentence in seen:
                continue
            selected.append(sentence)
            seen.add(sentence)
            if len(selected) >= 3:
                break

        if not selected:
            selected = [chunk["text"] for chunk in retrieved_chunks[:2] if chunk["text"]]

        answer = " ".join(selected).strip()
        answer = re.sub(r"\s+", " ", answer)
        if len(answer) > 500:
            answer = answer[:497].rsplit(" ", 1)[0] + "..."
        return answer

    @staticmethod
    def _trim_text(text: str, limit: int = 900) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _source_label(source: Any) -> str:
        if not source:
            return "Uploaded PDF"
        return Path(str(source)).name

    @staticmethod
    def _coerce_page_number(value: Any, zero_based: bool) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value + 1 if zero_based and value >= 0 else value
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed + 1 if zero_based else parsed

    @classmethod
    def _build_citations(cls, retrieved_chunks: list[dict[str, Any]]) -> list[str]:
        citations: list[str] = []
        for item in retrieved_chunks:
            page = item.get("page")
            source = item.get("source") or "Uploaded PDF"
            label = f"{source} p.{page}" if page else source
            if label not in citations:
                citations.append(label)
        return citations

    @classmethod
    def _build_sources(cls, retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen = set()
        for item in retrieved_chunks:
            source = item.get("source") or "Uploaded PDF"
            page = item.get("page")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": source, "page": page})
        return sources
