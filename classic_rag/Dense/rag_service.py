import re
from typing import List, Set

from classic_rag.Dense.rag_config import RAGResponse
from classic_rag.Dense.search_result import SearchResult


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

        if not filtered:
            filtered = reranked[:5]

        context = self._build_context(filtered)

        if len(context.strip()) < 30:
            context = self._fallback_context(reranked[:5])

        context = self._sanitize_context(context)

        raw_answer = self.generator.generate(
            query=query,
            context=context,
            hits=filtered
        )

        answer = self._validate_and_fix(raw_answer)

        sources = self._build_sources(filtered)

        if sources:
            answer = f"{answer}\n\nИсточник: {sources[0]}."

        return RAGResponse(
            answer=answer,
            sources=sources
        )

    def _sanitize_context(self, text: str) -> str:
        text = re.sub(r"(?i)\b(a:|q:)\b", "", text)
        text = re.sub(r"\bНедостаточно данных\b.*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _filter_hits(self, hits: List[SearchResult]) -> List[SearchResult]:

        filtered = []
        seen: Set[tuple] = set()

        for h in hits:

            article = h.payload.get("article_number")
            if not article:
                continue

            score = getattr(h, "final_score", 0.0)
            if score < self.min_final_score:
                continue

            header = (h.payload.get("header") or "").lower()

            key = (article, header)
            if key in seen:
                continue

            seen.add(key)
            filtered.append(h)

            if len(filtered) >= 6:
                break

        return filtered

    def _build_context(self, hits: List[SearchResult]) -> str:

        parts = []
        size = 0
        seen = set()

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < 30:
                continue

            norm = self._normalize(text)
            if norm in seen:
                continue

            seen.add(norm)

            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")

            block = f"""
            Статья {article} — {header}

            {text[:800]}
            """.strip()

            if size + len(block) > self.max_context_chars:
                break

            parts.append(block)
            size += len(block)

        return "\n\n".join(parts)

    def _fallback_context(self, hits: List[SearchResult]) -> str:

        parts = []

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < 50:
                continue

            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")

            parts.append(
                f"Статья {article} — {header}\n{text[:500]}"
            )

        return "\n\n".join(parts)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _validate_and_fix(self, text: str) -> str:

        if not text:
            return "Недостаточно данных."

        text = text.strip()

        text = re.sub(r"(?i)^(a:|q:)\s*", "", text)

        bad_patterns = [
            r"(?is)^(okay|let's|first|i need|the user).*?$",
            r"(?is)^(нужно ответить|сначала|важно|убеждаюсь).*?$",
            r"(?is)reasoning|analysis|explanation"
        ]

        for p in bad_patterns:
            text = re.sub(p, "", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        bullets = re.findall(r"(?:^|\n)-\s+(.*)", text)

        if bullets:
            return "Трудовое законодательство устанавливает " + \
                ", ".join(b.strip(" .") for b in bullets) + \
                " в соответствии с Трудовым кодексом РФ."

        text = re.sub(r"\s+", " ", text).strip()

        if len(text.split()) < 3:
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