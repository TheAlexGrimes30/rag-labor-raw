from typing import List

from classic_rag.Hybrid.rag_config import (
    RAGResponse
)
from classic_rag.Hybrid.search_result import SearchResult


class RAGService:

    def __init__(
            self,
            retriever,
            reranker,
            generator,
            max_context_chars: int = 7000
    ):

        self.retriever = retriever

        self.reranker = reranker

        self.generator = generator

        self.max_context_chars = max_context_chars

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(
            query=query,
            top_k=20
        )

        reranked = self.reranker.rerank(
            query=query,
            hits=hits,
            top_n=4
        )

        context = self._build_context(
            reranked
        )

        answer = self.generator.generate(
            query=query,
            context=context,
            hits=reranked
        )

        sources = self._build_sources(
            reranked
        )

        return RAGResponse(
            answer=answer,
            sources=sources
        )

    def _build_context(
            self,
            hits: List[SearchResult]
    ) -> str:

        parts = []

        current_size = 0

        seen = set()

        for h in hits:

            article = h.payload.get(
                "article_number",
                "?"
            )

            header = h.payload.get(
                "header",
                ""
            )

            text = (h.text or "").strip()

            if len(text) < 40:
                continue

            key = (
                str(article),
                header.lower()
            )

            if key in seen:
                continue

            seen.add(key)

            text = text[:1200]

            block = f"""
            Трудовой кодекс РФ
            Статья {article}
            Раздел: {header}
            
            {text}
            """.strip()

            if (
                    current_size + len(block)
                    > self.max_context_chars
            ):
                break

            parts.append(block)

            current_size += len(block)

        return "\n\n".join(parts)

    def _build_sources(
            self,
            hits: List[SearchResult]
    ) -> List[str]:

        seen = set()

        sources = []

        for h in hits:

            article = h.payload.get(
                "article_number"
            )

            if not article:
                continue

            source = (
                f"Трудовой кодекс РФ, статья {article}"
            )

            if source in seen:
                continue

            seen.add(source)

            sources.append(source)

        return sources