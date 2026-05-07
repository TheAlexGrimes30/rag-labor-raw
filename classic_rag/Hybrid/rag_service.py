from typing import List

from classic_rag.Hybrid.rag_config import RAGResponse, SearchResult


class RAGService:

    def __init__(
            self,
            retriever,
            reranker,
            generator,
            max_context_chars: int = 12000
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

        self.max_context_chars = max_context_chars

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(
            query,
            top_k=25
        )

        reranked = self.reranker.rerank(
            query=query,
            hits=hits,
            top_n=8
        )

        context = self._build_context(reranked)

        answer = self.generator.generate(
            query=query,
            context=context
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

        for i, h in enumerate(hits, start=1):

            article = h.payload.get(
                "article_number",
                "?"
            )

            header = h.payload.get(
                "header",
                ""
            )

            score = round(
                float(h.score),
                4
            )

            text = (h.text or "").strip()

            if len(text) < 30:
                continue

            block = f"""
            [Источник #{i}]
            [Релевантность: {score}]

            Статья: {article}
            Раздел: {header}

            {text}
            """.strip()

            if current_size + len(block) > self.max_context_chars:
                break

            parts.append(block)

            current_size += len(block)

        return "\n\n-------------------\n\n".join(parts)

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

            header = h.payload.get(
                "header",
                ""
            )

            if not article:
                continue

            key = (article, header)

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                f"Статья {article} — {header}"
            )

        return sources
