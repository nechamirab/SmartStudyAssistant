from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Any


class ContextRetrievalService:
    """Local lexical retrieval for selecting PDF context before AI calls."""

    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
        "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "in", "into", "is", "it", "its", "of", "on", "or", "should",
        "that", "the", "their", "there", "these", "this", "those", "to",
        "was", "were", "what", "when", "where", "which", "why", "with",
        "would", "explain", "describe", "define", "tell", "about",
        "give", "me", "please", "show", "using", "use",
    }
    BM25_K1 = 1.5
    BM25_B = 0.75
    DEFAULT_SCORE_THRESHOLD = 1.0

    @classmethod
    def detect_query_intent(cls, question: str) -> dict[str, Any]:
        raw_question = question or ""
        normalized = raw_question.lower()
        chapter_numbers = cls._extract_numbered_references(normalized, ["chapter", "chapters", "ch", "פרק", "פרקים"])
        section_numbers = cls._extract_numbered_references(normalized, ["section", "sections", "חלק"])
        wants_summary = bool(re.search(r"\b(summarize|summary|explain|overview)\b|סכם|סיכום", normalized))
        wants_main_idea = bool(
            re.search(r"\b(main idea|main ideas|key idea|key ideas|main point|main points|about)\b", normalized)
            or "רעיון מרכזי" in normalized
        )
        wants_study_plan = bool(
            re.search(r"\b(study plan|prepare for|help me study)\b", normalized)
            or "תוכנית לימוד" in normalized
        )
        mentions_pdf_summary = bool(
            re.search(r"\b(pdf|document|material)\b", normalized)
            and (wants_summary or wants_main_idea)
        ) or any(phrase in normalized for phrase in ["מהמסמך", "מהחומר"])

        if chapter_numbers and (wants_summary or wants_main_idea or "chapter" in normalized or "פרק" in normalized):
            intent = "chapter_summary"
        elif section_numbers and (wants_summary or wants_main_idea or "section" in normalized or "חלק" in normalized):
            intent = "section_summary"
        elif wants_study_plan:
            intent = "study_plan"
        elif mentions_pdf_summary:
            intent = "general_pdf_summary"
        elif raw_question.strip():
            intent = "factual_question"
        else:
            intent = "unknown"

        return {
            "intent": intent,
            "chapter_numbers": chapter_numbers,
            "section_numbers": section_numbers,
            "wants_summary": wants_summary,
            "wants_main_idea": wants_main_idea,
            "raw_question": raw_question,
        }

    @classmethod
    def build_chunks_from_pages(cls, pages: list[Any], sections: list[Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        sorted_sections = sorted(
            sections or [],
            key=lambda section: int(getattr(section, "section_number", 0) or 0),
        )

        for page in pages or []:
            page_number = int(getattr(page, "page_number", 0) or 0)
            page_text = cls._clean_text(getattr(page, "text", "") or "")
            if not page_text:
                continue

            section = cls._section_for_page(page_number, sorted_sections)
            if section is None:
                section_number = 0
                section_title = "Unassigned section"
                start_page = page_number
                end_page = page_number
                key_concepts: list[str] = []
            else:
                section_number = int(getattr(section, "section_number", 0) or 0)
                section_title = str(getattr(section, "title", "") or f"Section {section_number}")
                start_page = int(getattr(section, "start_page", page_number) or page_number)
                end_page = int(getattr(section, "end_page", page_number) or page_number)
                key_concepts = list(getattr(section, "key_concepts", []) or [])

            for text_part in cls._split_text(page_text):
                chunks.append(
                    {
                        "section_number": section_number,
                        "section_title": section_title,
                        "start_page": start_page,
                        "end_page": end_page,
                        "page": page_number,
                        "text": text_part,
                        "key_concepts": key_concepts,
                    }
                )

        return chunks

    @classmethod
    def retrieve_relevant_chunks(
        cls,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        min_score: int = 1,
    ) -> list[dict[str, Any]]:
        intent = cls.detect_query_intent(question)
        query = cls._analyze_query(question)
        if not query["terms"]:
            return []

        indexed_chunks = cls._build_retrieval_index(chunks or [])
        if not indexed_chunks:
            return []

        threshold = max(cls.DEFAULT_SCORE_THRESHOLD, float(min_score or 0))
        idf = cls._idf_by_term(indexed_chunks)
        avgdl = sum(max(1, item["text_length"]) for item in indexed_chunks) / max(1, len(indexed_chunks))

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, item in enumerate(indexed_chunks):
            chunk = item["chunk"]
            score, diagnostics = cls._score_chunk(
                chunk=item,
                query=query,
                idf=idf,
                avgdl=avgdl,
                intent=intent,
                threshold=threshold,
            )

            if not diagnostics["accepted"]:
                continue

            ranked_chunk = dict(chunk)
            ranked_chunk["score"] = round(score, 4)
            ranked_chunk["matched_meaningful_tokens"] = diagnostics["matched_meaningful_tokens"]
            ranked_chunk["unmatched_query_tokens"] = diagnostics["unmatched_query_tokens"]
            ranked_chunk["ignored_query_tokens"] = diagnostics["ignored_query_tokens"]
            ranked_chunk["phrase_matches"] = diagnostics["phrase_matches"]
            ranked_chunk["threshold_used"] = threshold
            ranked_chunk["passed_by"] = diagnostics["passed_by"]
            ranked_chunk["match_reason"] = ", ".join(diagnostics["passed_by"]) or "lexical match"
            scored.append((score, index, ranked_chunk))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[: max(1, int(top_k or 1))]]

    @classmethod
    def _analyze_query(cls, question: str) -> dict[str, Any]:
        raw_tokens = [
            token
            for token in re.findall(r"[a-z0-9\u0590-\u05ff]+", (question or "").lower())
            if len(token) > 1
        ]
        meaningful_tokens = cls._token_list(question)
        ignored_tokens = [token for token in raw_tokens if token not in meaningful_tokens]
        terms: set[str] = set()
        term_variants_by_token: dict[str, set[str]] = {}

        for token in meaningful_tokens:
            variants = cls._lexical_variants(token)
            term_variants_by_token[token] = variants
            terms.update(variants)

        return {
            "meaningful_tokens": meaningful_tokens,
            "ignored_tokens": sorted(dict.fromkeys(ignored_tokens)),
            "terms": terms,
            "term_variants_by_token": term_variants_by_token,
            "phrases": cls._query_phrases(meaningful_tokens),
        }

    @classmethod
    def _build_retrieval_index(cls, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        indexed: list[dict[str, Any]] = []
        for chunk in chunks:
            text_tokens = cls._index_token_list(str(chunk.get("text", "") or ""))
            title_tokens = cls._index_token_list(str(chunk.get("section_title", "") or ""))
            concept_tokens = cls._index_token_list(" ".join(str(item) for item in chunk.get("key_concepts", []) or []))
            indexed.append(
                {
                    "chunk": chunk,
                    "text_counts": Counter(text_tokens),
                    "title_counts": Counter(title_tokens),
                    "concept_counts": Counter(concept_tokens),
                    "all_terms": set(text_tokens) | set(title_tokens) | set(concept_tokens),
                    "text_length": len(text_tokens),
                    "normalized_text": " ".join(cls._token_list(str(chunk.get("text", "") or ""))),
                    "normalized_title": " ".join(cls._token_list(str(chunk.get("section_title", "") or ""))),
                    "normalized_concepts": " ".join(
                        cls._token_list(" ".join(str(item) for item in chunk.get("key_concepts", []) or []))
                    ),
                }
            )
        return indexed

    @staticmethod
    def _idf_by_term(indexed_chunks: list[dict[str, Any]]) -> dict[str, float]:
        document_count = max(1, len(indexed_chunks))
        document_frequency: Counter[str] = Counter()
        for item in indexed_chunks:
            document_frequency.update(item["all_terms"])
        return {
            term: log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def _score_chunk(
        cls,
        chunk: dict[str, Any],
        query: dict[str, Any],
        idf: dict[str, float],
        avgdl: float,
        intent: dict[str, Any],
        threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        score = 0.0
        matched_terms: set[str] = set()
        passed_by: list[str] = []

        for term in query["terms"]:
            text_tf = int(chunk["text_counts"].get(term, 0))
            title_tf = int(chunk["title_counts"].get(term, 0))
            concept_tf = int(chunk["concept_counts"].get(term, 0))
            if not text_tf and not title_tf and not concept_tf:
                continue

            weight = max(0.1, idf.get(term, 0.1))
            if text_tf:
                score += cls._bm25_term_score(text_tf, chunk["text_length"], avgdl, weight)
            if title_tf:
                score += weight * 2.5
            if concept_tf:
                score += weight * 2.0
            matched_terms.add(term)

        phrase_matches = cls._phrase_matches(query["phrases"], chunk)
        if phrase_matches:
            for phrase in phrase_matches:
                phrase_terms = phrase.split()
                score += sum(max(0.1, idf.get(term, 0.1)) for term in phrase_terms) * 3.0
                matched_terms.update(phrase_terms)
            passed_by.append("phrase boost")

        section_number = int(chunk["chunk"].get("section_number", 0) or 0)
        metadata_match = False
        if section_number and section_number in intent.get("section_numbers", []):
            score += 10.0
            metadata_match = True
            passed_by.append("requested section number")
        if section_number and section_number in intent.get("chapter_numbers", []):
            score += 8.0
            metadata_match = True
            passed_by.append("requested chapter mapped to section")

        matched_meaningful_tokens = cls._matched_meaningful_tokens(query, matched_terms)
        unmatched_query_tokens = [
            token
            for token in query["meaningful_tokens"]
            if token not in matched_meaningful_tokens
        ]

        enough_terms = cls._has_enough_term_evidence(
            meaningful_tokens=query["meaningful_tokens"],
            matched_meaningful_tokens=matched_meaningful_tokens,
            score=score,
            threshold=threshold,
            phrase_matches=phrase_matches,
            metadata_match=metadata_match,
        )
        accepted = score >= threshold and enough_terms
        if accepted and "phrase boost" not in passed_by and not metadata_match:
            passed_by.append("bm25 score")

        return score, {
            "accepted": accepted,
            "matched_meaningful_tokens": matched_meaningful_tokens,
            "unmatched_query_tokens": unmatched_query_tokens,
            "ignored_query_tokens": query["ignored_tokens"],
            "phrase_matches": phrase_matches,
            "passed_by": passed_by,
        }

    @classmethod
    def _bm25_term_score(cls, term_frequency: int, document_length: int, avgdl: float, idf: float) -> float:
        normalized_length = cls.BM25_K1 * (
            1 - cls.BM25_B + cls.BM25_B * (max(1, document_length) / max(1.0, avgdl))
        )
        return idf * ((term_frequency * (cls.BM25_K1 + 1)) / (term_frequency + normalized_length))

    @classmethod
    def _matched_meaningful_tokens(cls, query: dict[str, Any], matched_terms: set[str]) -> list[str]:
        matched: list[str] = []
        for token in query["meaningful_tokens"]:
            variants = query["term_variants_by_token"].get(token, {token})
            if variants & matched_terms:
                matched.append(token)
        return matched

    @staticmethod
    def _has_enough_term_evidence(
        meaningful_tokens: list[str],
        matched_meaningful_tokens: list[str],
        score: float,
        threshold: float,
        phrase_matches: list[str],
        metadata_match: bool,
    ) -> bool:
        if metadata_match:
            return True
        if not meaningful_tokens or not matched_meaningful_tokens:
            return False
        if phrase_matches:
            return True
        if len(matched_meaningful_tokens) >= 2:
            return True
        if len(meaningful_tokens) == 1:
            return score >= threshold
        return score >= max(threshold * 2.5, 2.5)

    @staticmethod
    def _phrase_matches(phrases: list[str], chunk: dict[str, Any]) -> list[str]:
        haystacks = [
            chunk["normalized_text"],
            chunk["normalized_title"],
            chunk["normalized_concepts"],
        ]
        matches: list[str] = []
        for phrase in phrases:
            phrase_pattern = f" {phrase} "
            if any(phrase_pattern in f" {haystack} " for haystack in haystacks):
                matches.append(phrase)
        return matches

    @staticmethod
    def _query_phrases(tokens: list[str]) -> list[str]:
        phrases: list[str] = []
        for size in range(min(4, len(tokens)), 1, -1):
            for index in range(0, len(tokens) - size + 1):
                phrase_tokens = tokens[index : index + size]
                if any(token.isdigit() for token in phrase_tokens):
                    continue
                phrases.append(" ".join(phrase_tokens))
        return phrases

    @staticmethod
    def retrieve_overview_chunks(chunks: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_sections: set[int] = set()

        for chunk in chunks or []:
            section_number = int(chunk.get("section_number", 0) or 0)
            if section_number in seen_sections:
                continue
            if not str(chunk.get("text", "") or "").strip():
                continue
            overview_chunk = dict(chunk)
            overview_chunk["score"] = 0
            selected.append(overview_chunk)
            seen_sections.add(section_number)
            if len(selected) >= max(1, int(top_k or 1)):
                return selected

        for chunk in chunks or []:
            if len(selected) >= max(1, int(top_k or 1)):
                break
            if not str(chunk.get("text", "") or "").strip():
                continue
            overview_chunk = dict(chunk)
            overview_chunk["score"] = 0
            selected.append(overview_chunk)

        return selected

    @classmethod
    def build_document_index(cls, pages: list[Any], sections: list[Any]) -> dict[str, list[dict[str, Any]]]:
        ordered_pages = sorted(
            [
                {
                    "page_number": int(getattr(page, "page_number", 0) or 0),
                    "text": cls._clean_text(getattr(page, "text", "") or ""),
                }
                for page in pages or []
                if (getattr(page, "text", "") or "").strip()
            ],
            key=lambda item: item["page_number"],
        )
        ordered_sections = sorted(
            sections or [],
            key=lambda section: int(getattr(section, "section_number", 0) or 0),
        )
        indexed_sections = [
            {
                "section_number": int(getattr(section, "section_number", 0) or 0),
                "title": str(getattr(section, "title", "") or f"Section {int(getattr(section, 'section_number', 0) or 0)}"),
                "start_page": int(getattr(section, "start_page", 0) or 0),
                "end_page": int(getattr(section, "end_page", 0) or 0),
                "summary": str(getattr(section, "summary", "") or ""),
                "key_concepts": list(getattr(section, "key_concepts", []) or []),
                "estimated_minutes": int(getattr(section, "estimated_minutes", 0) or 0),
                "difficulty": str(getattr(section, "difficulty", "") or ""),
                "text": cls._text_for_page_range(
                    ordered_pages,
                    int(getattr(section, "start_page", 0) or 0),
                    int(getattr(section, "end_page", 0) or 0),
                ),
            }
            for section in ordered_sections
        ]

        heading_matches = cls._detect_chapter_headings(ordered_pages)
        chapters: list[dict[str, Any]] = []
        if heading_matches:
            for index, heading in enumerate(heading_matches):
                start_page = int(heading["start_page"])
                next_start = int(heading_matches[index + 1]["start_page"]) if index + 1 < len(heading_matches) else None
                end_page = (next_start - 1) if next_start is not None else (ordered_pages[-1]["page_number"] if ordered_pages else start_page)
                chapters.append(
                    {
                        "chapter_number": int(heading["chapter_number"]),
                        "title": str(heading["title"]),
                        "start_page": start_page,
                        "end_page": max(start_page, end_page),
                        "text": cls._text_for_page_range(ordered_pages, start_page, max(start_page, end_page)),
                        "matched_by": "heading",
                    }
                )
        else:
            chapters = [
                {
                    "chapter_number": item["section_number"],
                    "title": item["title"],
                    "start_page": item["start_page"],
                    "end_page": item["end_page"],
                    "text": item["text"],
                    "matched_by": "section_fallback",
                }
                for item in indexed_sections
                if item["section_number"]
            ]

        return {"chapters": chapters, "sections": indexed_sections, "pages": ordered_pages}

    @classmethod
    def retrieve_chapter_context(
        cls,
        chapter_numbers: list[int],
        pages: list[Any],
        sections: list[Any],
        max_chars: int = 9000,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not chapter_numbers:
            return "", []
        index = cls.build_document_index(pages, sections)
        chapter_by_number = {int(item["chapter_number"]): item for item in index["chapters"]}
        requested = [chapter_by_number[number] for number in chapter_numbers if number in chapter_by_number]
        if not requested:
            return "", []

        per_item_chars = max(600, max_chars // max(1, len(requested)) - 160)
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        used = 0
        for chapter in requested:
            excerpt = cls._meaningful_excerpt(str(chapter.get("text", "") or ""), per_item_chars)
            if not excerpt:
                continue
            start_page = int(chapter["start_page"])
            end_page = int(chapter["end_page"])
            number = int(chapter["chapter_number"])
            title = str(chapter["title"])
            header = f"[Chapter {number} | {title} | Pages {start_page}-{end_page}]"
            block = f"{header}\n{excerpt}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rsplit(" ", 1)[0].rstrip()
            if not block:
                continue
            parts.append(block)
            used += len(block) + 2
            source_type = "chapter" if chapter.get("matched_by") == "heading" else "study_section"
            sources.append(
                {
                    "type": source_type,
                    "number": number,
                    "title": title,
                    "start_page": start_page,
                    "end_page": end_page,
                    "matched_by": chapter.get("matched_by", ""),
                }
            )

        return "\n\n".join(parts).strip(), sources

    @staticmethod
    def format_chunks_for_prompt(chunks: list[dict[str, Any]], max_chars: int = 7000) -> str:
        parts: list[str] = []
        used = 0

        for chunk in chunks or []:
            header = (
                f"[Section {int(chunk.get('section_number', 0) or 0)} | "
                f"{chunk.get('section_title', 'Untitled section')} | "
                f"Page {int(chunk.get('page', 0) or 0)}]"
            )
            text = re.sub(r"\s+", " ", str(chunk.get("text", "") or "")).strip()
            if not text:
                continue

            block = f"{header}\n{text}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rsplit(" ", 1)[0].rstrip()
            if block:
                parts.append(block)
                used += len(block) + 2

        return "\n\n".join(parts).strip()

    @classmethod
    def retrieve_exam_context(cls, sections: list[Any], pages: list[Any], max_chars: int = 12000) -> str:
        if not sections or not pages or max_chars <= 0:
            return ""

        parts: list[str] = []
        used = 0
        ordered_sections = sorted(
            sections,
            key=lambda section: int(getattr(section, "section_number", 0) or 0),
        )
        section_count = max(1, len(ordered_sections))
        per_section_excerpt = max(300, min(1200, max_chars // section_count - 250))

        for section in ordered_sections:
            section_number = int(getattr(section, "section_number", 0) or 0)
            title = str(getattr(section, "title", "") or f"Section {section_number}")
            start_page = int(getattr(section, "start_page", 0) or 0)
            end_page = int(getattr(section, "end_page", start_page) or start_page)
            summary = str(getattr(section, "summary", "") or "").strip()
            concepts = ", ".join(str(item) for item in (getattr(section, "key_concepts", []) or [])[:8])
            section_text = cls._section_excerpt(pages, start_page, end_page, per_section_excerpt)

            block = "\n".join(
                item
                for item in [
                    f"Section {section_number}: {title}",
                    f"Pages: {start_page}-{end_page}" if start_page != end_page else f"Page: {start_page}",
                    f"Summary: {summary}" if summary else "",
                    f"Key concepts: {concepts}" if concepts else "",
                    f"Representative text: {section_text}" if section_text else "",
                ]
                if item
            )

            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining].rsplit(" ", 1)[0].rstrip()
            if block:
                parts.append(block)
                used += len(block) + 2

        return "\n\n".join(parts).strip()

    @classmethod
    def source_labels(cls, chunks: list[dict[str, Any]]) -> list[str]:
        labels: list[str] = []
        seen: set[tuple[int, int]] = set()
        for chunk in chunks or []:
            section_number = int(chunk.get("section_number", 0) or 0)
            page = int(chunk.get("page", 0) or 0)
            key = (section_number, page)
            if key in seen:
                continue
            seen.add(key)
            title = str(chunk.get("section_title", "") or "Untitled section")
            labels.append(f"Section {section_number}, Page {page}: {title}")
        return labels

    @classmethod
    def _token_list(cls, value: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9\u0590-\u05ff]+", (value or "").lower())
            if len(token) > 1 and token not in cls.STOP_WORDS
        ]

    @classmethod
    def _index_token_list(cls, value: str) -> list[str]:
        tokens: list[str] = []
        for token in cls._token_list(value):
            tokens.extend(sorted(cls._lexical_variants(token)))
        return tokens

    @classmethod
    def _lexical_variants(cls, token: str) -> set[str]:
        variants = {token}
        if token.endswith("s") and len(token) > 3:
            variants.add(token[:-1])
        elif len(token) > 2 and re.fullmatch(r"[a-z]+", token):
            variants.add(token + "s")
        if token in cls.NUMBER_WORDS:
            variants.add(str(cls.NUMBER_WORDS[token]))
        elif token.isdigit():
            for word, number in cls.NUMBER_WORDS.items():
                if number == int(token):
                    variants.add(word)
        return variants

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        return set(cls._token_list(value))

    @classmethod
    def _expanded_tokens(cls, value: str) -> set[str]:
        tokens = cls._tokens(value)
        expanded = set(tokens)
        expansions = {
            "learn": {"learning", "learned"},
            "learning": {"learn"},
            "memory": {"memories"},
            "memories": {"memory"},
            "explain": {"explanation"},
            "explanation": {"explain"},
            "summarize": {"summary"},
            "summary": {"summarize", "main", "idea", "key"},
            "main": {"summary", "key"},
            "idea": {"summary", "point"},
            "ideas": {"summary", "idea", "points"},
            "chapter": {"section", "topic"},
            "section": {"chapter", "topic"},
            "topic": {"chapter", "section"},
        }
        for token in list(tokens):
            expanded.update(expansions.get(token, set()))
            if token.endswith("s") and len(token) > 3:
                expanded.add(token[:-1])
            elif len(token) > 2:
                expanded.add(token + "s")
            if token in cls.NUMBER_WORDS:
                expanded.add(str(cls.NUMBER_WORDS[token]))
            elif token.isdigit():
                for word, number in cls.NUMBER_WORDS.items():
                    if number == int(token):
                        expanded.add(word)
        return expanded

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def _split_text(cls, text: str, max_chars: int = 1400, overlap: int = 160) -> list[str]:
        text = cls._clean_text(text)
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                split_at = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if split_at > start + max_chars // 2:
                    end = split_at + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _section_for_page(page_number: int, sections: list[Any]) -> Any | None:
        for section in sections:
            start_page = int(getattr(section, "start_page", 0) or 0)
            end_page = int(getattr(section, "end_page", start_page) or start_page)
            if start_page <= page_number <= end_page:
                return section
        return None

    @classmethod
    def _section_excerpt(cls, pages: list[Any], start_page: int, end_page: int, max_chars: int) -> str:
        text = " ".join(
            getattr(page, "text", "") or ""
            for page in pages
            if start_page <= int(getattr(page, "page_number", 0) or 0) <= end_page
        )
        return cls._clean_text(text)[:max_chars].rstrip()

    @classmethod
    def _extract_numbered_references(cls, normalized: str, labels: list[str]) -> list[int]:
        numbers: list[int] = []
        label_pattern = "|".join(re.escape(label) for label in labels)
        number_pattern = r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
        for match in re.finditer(rf"\b(?:{label_pattern})\.?\s+({number_pattern})(?:\s*(?:-|to|and|&)\s*({number_pattern}))?", normalized):
            first = cls._parse_number(match.group(1))
            second = cls._parse_number(match.group(2)) if match.group(2) else None
            if first is None:
                continue
            if second is not None and second > first and second - first <= 10:
                numbers.extend(range(first, second + 1))
            else:
                numbers.append(first)
                if second is not None:
                    numbers.append(second)

        hebrew_label = any(label in {"פרק", "פרקים", "חלק"} for label in labels)
        if hebrew_label:
            for match in re.finditer(r"(?:פרק|פרקים|חלק)\s+(\d{1,2})(?:\s*(?:-|ו|and)\s*(\d{1,2}))?", normalized):
                first = cls._parse_number(match.group(1))
                second = cls._parse_number(match.group(2)) if match.group(2) else None
                if first is not None:
                    numbers.append(first)
                if second is not None:
                    numbers.append(second)
        return sorted(dict.fromkeys(number for number in numbers if 0 < number <= 99))

    @classmethod
    def _parse_number(cls, value: str | None) -> int | None:
        if not value:
            return None
        value = value.lower().strip()
        if value.isdigit():
            return int(value)
        return cls.NUMBER_WORDS.get(value)

    @classmethod
    def _detect_chapter_headings(cls, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        headings: list[dict[str, Any]] = []
        seen: set[int] = set()
        patterns = [
            re.compile(r"^(chapter|ch\.?)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b[:.\-\s]*(.*)", re.I),
            re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z][^\n]{2,80})"),
            re.compile(r"^(\d{1,2})\s+([A-Z][A-Za-z][^\n]{2,80})"),
        ]
        for page in pages:
            page_number = int(page["page_number"])
            raw_lines = re.split(r"\n+|(?<=\.)\s{2,}", page.get("text", ""))
            page_start = page.get("text", "")[:900]
            candidates = [line.strip() for line in raw_lines[:12] if line.strip()]
            if not candidates and page_start:
                candidates = [page_start.strip()]
            for line in candidates:
                line = re.sub(r"\s+", " ", line).strip()[:140]
                match_data = cls._match_chapter_heading(line, patterns)
                if not match_data:
                    continue
                number, title = match_data
                if number in seen:
                    continue
                seen.add(number)
                headings.append(
                    {
                        "chapter_number": number,
                        "title": title or f"Chapter {number}",
                        "start_page": page_number,
                    }
                )
                break
        headings.sort(key=lambda item: (item["start_page"], item["chapter_number"]))
        return headings

    @classmethod
    def _match_chapter_heading(cls, line: str, patterns: list[re.Pattern[str]]) -> tuple[int, str] | None:
        if len(line.split()) > 14:
            return None
        for index, pattern in enumerate(patterns):
            match = pattern.match(line)
            if not match:
                continue
            if index == 0:
                number = cls._parse_number(match.group(2))
                suffix = (match.group(3) or "").strip(" :-")
                if number is None:
                    return None
                title = f"Chapter {number}" + (f": {suffix}" if suffix else "")
                return number, title
            number = cls._parse_number(match.group(1))
            suffix = (match.group(2) or "").strip(" :-")
            if number is None:
                return None
            return number, f"Chapter {number}: {suffix}" if suffix else f"Chapter {number}"
        return None

    @staticmethod
    def _text_for_page_range(pages: list[dict[str, Any]], start_page: int, end_page: int) -> str:
        return " ".join(
            page["text"]
            for page in pages
            if start_page <= int(page["page_number"]) <= end_page and page.get("text")
        ).strip()

    @classmethod
    def _meaningful_excerpt(cls, text: str, max_chars: int) -> str:
        text = cls._clean_text(text)
        if len(text) <= max_chars:
            return text
        part_chars = max(180, max_chars // 3)
        start = text[:part_chars].rsplit(" ", 1)[0].rstrip()
        middle_start = max(0, len(text) // 2 - part_chars // 2)
        middle = text[middle_start : middle_start + part_chars].strip()
        middle = middle.split(" ", 1)[-1].rsplit(" ", 1)[0].strip()
        end = text[-part_chars:].split(" ", 1)[-1].strip()
        return f"{start}\n...\n{middle}\n...\n{end}".strip()
