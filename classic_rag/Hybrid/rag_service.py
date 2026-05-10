import re
from typing import List, Set

from classic_rag.Hybrid.rag_config import RAGResponse
from classic_rag.Hybrid.search_result import SearchResult


class RAGService:

    def __init__(
            self,
            retriever,
            reranker,
            generator,
            max_context_chars: int = 3500,
            min_final_score: float = 0.50
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

        self.max_context_chars = max_context_chars
        self.min_final_score = min_final_score

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(query=query, top_k=25)

        reranked = self.reranker.rerank(query=query, hits=hits, top_n=10)

        filtered = self._filter_hits(reranked)

        if len(filtered) == 0:
            filtered = reranked[:5]

        # 3. CONTEXT BUILD
        context = self._build_context(filtered)

        if not context or len(context.strip()) < 30:
            context = self._fallback_context(reranked[:5])

        raw_answer = self.generator.generate(
            query=query,
            context=context,
            hits=filtered
        )

        answer = self._validate_and_fix(raw_answer)

        sources = self._build_sources(filtered)

        return RAGResponse(
            answer=answer,
            sources=sources
        )

    def _filter_hits(self, hits: List[SearchResult]) -> List[SearchResult]:

        filtered = []
        seen: Set[tuple] = set()

        for h in hits:

            article = h.payload.get("article_number")
            header = (h.payload.get("header") or "").lower()

            if not article:
                continue

            if self._is_noise_header(header):
                continue

            score = getattr(h, "final_score", 0.0)

            if score < self.min_final_score:
                continue

            key = (article, header)
            if key in seen:
                continue

            seen.add(key)
            filtered.append(h)

            if len(filtered) >= 6:
                break

        return filtered

    def _is_noise_header(self, header: str) -> bool:

        noise_patterns = [
            "краткое содержание",
            "практическое значение",
            "общие выводы",
            "комментарий",
            "введение",
            "систематизация"
        ]

        safe_headers = [
            "основные",
            "сфера действия",
            "обязанности",
            "понятие",
            "цели",
            "задачи"
        ]

        if any(s in header for s in safe_headers):
            return False

        return any(p in header for p in noise_patterns)

    def _build_context(self, hits: List[SearchResult]) -> str:

        parts = []
        size = 0
        seen = set()

        for h in hits:

            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")
            text = (h.text or "").strip()

            if len(text) < 20:
                continue

            norm = self._normalize(text)
            if norm in seen:
                continue

            seen.add(norm)

            block = f"""Статья {article} — {header}

{text[:900]}
""".strip()

            if size + len(block) > self.max_context_chars:
                break

            parts.append(block)
            size += len(block)

        return "\n\n".join(parts)

    def _fallback_context(self, hits: List[SearchResult]) -> str:

        parts = []

        for h in hits:

            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")
            text = (h.text or "").strip()

            parts.append(
                f"Статья {article} — {header}\n\n{text[:700]}"
            )

        return "\n\n".join(parts)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _validate_and_fix(self, text: str) -> str:

        if not text:
            return "Недостаточно данных."

        text = text.strip()

        bad_tokens = [
            "A:", "Q:",
            "reasoning", "explanation",
            "обоснование", "анализ"
        ]

        lower = text.lower()

        cut_pos = len(text)
        for token in bad_tokens:
            idx = lower.find(token)
            if idx != -1:
                cut_pos = min(cut_pos, idx)

        text = text[:cut_pos].strip()

        lines = []
        for line in text.splitlines():
            l = line.strip()
            if not l:
                continue
            if any(x in l.lower() for x in bad_tokens):
                continue
            lines.append(l)

        text = "\n".join(lines).strip()

        if len(text) < 5:
            return "Недостаточно данных."

        return text

    def _build_sources(self, hits: List[SearchResult]) -> List[str]:

        seen = set()
        out = []

        for h in hits:

            article = h.payload.get("article_number")
            if not article:
                continue

            src = f"Трудовой кодекс РФ, статья {article}"

            if src not in seen:
                seen.add(src)
                out.append(src)

        return out