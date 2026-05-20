from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

import math
import re

from sentence_transformers import CrossEncoder

from classic_rag.Dense.search_result import SearchResult


class BaseReranker(ABC):
    """
    Abstract base interface for reranking systems.

    A reranker receives:
    - user query
    - retrieved search results

    and returns:
    - reordered search results ranked by relevance.
    """


    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: list["SearchResult"],
        *,
        top_n: int
    ) -> list["SearchResult"]:
        """
        Rerank retrieved search results.

        Args:
            query (str):
                User search query.

            hits (List[SearchResult]):
                Retrieved search results from retriever.

            top_n (int):
                Number of final results to return.

        Returns:
            List[SearchResult]:
                Reranked search results sorted by relevance.
        """

        raise NotImplementedError


class Reranker(BaseReranker):
    """
    Hybrid semantic reranker for legal RAG systems.

    Combines:
    - CrossEncoder semantic reranking
    - dense retriever score
    - lexical overlap scoring
    - header relevance boosting
    - definition-aware boosting
    - heuristic penalties
    - diversity filtering

    Designed specifically for legal and нормативный retrieval.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 8,
        max_length: int = 768,
        top_n: int = 5,

        rerank_weight: float = 0.40,
        dense_weight: float = 0.45,
        lexical_weight: float = 0.15,

        exact_header_boost: float = 0.20,
        partial_header_boost: float = 0.10,
        definition_boost: float = 0.20,

        generic_header_penalty: float = 0.03,
        low_lexical_penalty: float = 0.02,

        max_chunks_per_article: int = 3,
    ):
        """
        Initialize reranker.

        Args:
            model_name (str):
                HuggingFace CrossEncoder model name.

            batch_size (int):
                Batch size for inference.

            max_length (int):
                Maximum token length for model input.

            top_n (int):
                Default number of returned results.

            rerank_weight (float):
                Weight of CrossEncoder rerank score.

            dense_weight (float):
                Weight of dense retriever similarity score.

            lexical_weight (float):
                Weight of lexical overlap score.

            exact_header_boost (float):
                Boost for exact header match.

            partial_header_boost (float):
                Boost for partial header match.

            definition_boost (float):
                Additional boost for definitional chunks.

            generic_header_penalty (float):
                Penalty for generic headers.

            low_lexical_penalty (float):
                Penalty for low lexical overlap.

            max_chunks_per_article (int):
                Maximum chunks allowed from same article.

        Returns:
            None
        """

        self.model = self._load(
            model_name=model_name,
            max_length=max_length
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.top_n = top_n

        self.rerank_weight = rerank_weight
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        self.exact_header_boost = exact_header_boost
        self.partial_header_boost = partial_header_boost
        self.definition_boost = definition_boost

        self.generic_header_penalty = generic_header_penalty
        self.low_lexical_penalty = low_lexical_penalty

        self.max_chunks_per_article = max_chunks_per_article



    @staticmethod
    @lru_cache(maxsize=1)
    def _load(
        model_name: str,
        max_length: int
    ) -> CrossEncoder:
        """
        Load CrossEncoder model with caching.

        Args:
            model_name (str):
                HuggingFace model identifier.

            max_length (int):
                Maximum sequence length.

        Returns:
            CrossEncoder:
                Loaded reranker model.
        """

        model = CrossEncoder(
            model_name,
            max_length=max_length
        )

        tokenizer = model.tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.model.config.pad_token_id = (
            tokenizer.pad_token_id
        )

        return model


    def rerank(
        self,
        query: str,
        hits: list[SearchResult],
        top_n: int | None = None
    ) -> list[SearchResult]:
        """
        Perform hybrid reranking.

        Combines:
        - semantic reranking
        - dense score
        - lexical similarity
        - heuristic boosts and penalties

        Args:
            query (str):
                User search query.

            hits (List[SearchResult]):
                Retrieved search results.

            top_n (int | None):
                Number of results to return.

        Returns:
            List[SearchResult]:
                Final reranked search results.
        """

        if not hits:
            return []

        query = (query or "").strip()

        if not query:
            return []

        top_n = top_n or self.top_n

        valid_hits = [
            h for h in hits
            if h.text and h.text.strip()
        ]

        if not valid_hits:
            return []

        pairs = [
            self._build_pair(query, h)
            for h in valid_hits
        ]

        try:

            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

        except Exception as e:

            print(f"[RERANK ERROR] {e}")

            return sorted(
                valid_hits,
                key=lambda x: getattr(x, "score", 0.0),
                reverse=True
            )[:top_n]

        scored_hits: List[SearchResult] = []

        for hit, raw_score in zip(
            valid_hits,
            raw_scores
        ):

            rerank_score = self._normalize_logit(
                raw_score
            )

            dense_score = self._normalize_dense(
                getattr(hit, "score", 0.0)
            )

            lexical_score = self._lexical_score(
                query=query,
                text=hit.text or ""
            )

            payload = hit.payload or {}

            header = payload.get(
                "header",
                ""
            )

            header_score = self._header_score(
                query=query,
                header=header
            )

            definition_score = self._definition_score(
                query=query,
                text=hit.text or "",
                header=header
            )

            penalty = self._penalty_score(
                query=query,
                header=header,
                text=hit.text or ""
            )

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                header_score +
                definition_score -
                penalty
            )

            scored_hits.append(
                SearchResult.from_rerank(
                    base=hit,
                    score=final_score
                )
            )

        scored_hits.sort(
            key=lambda x: x.score,
            reverse=True
        )

        diversified = self._diversify(
            scored_hits,
            top_n=top_n
        )

        return diversified[:top_n]


    def _build_pair(
        self,
        query: str,
        doc: SearchResult
    ) -> tuple[str, str]:
        """
        Build query-document pair for CrossEncoder.

        Args:
            query (str):
                User query.

            doc (SearchResult):
                Retrieved document chunk.

        Returns:
            Tuple[str, str]:
                Pair of query and enriched document text.
        """

        payload = doc.payload or {}

        article = payload.get(
            "article_number",
            ""
        )

        header = payload.get(
            "header",
            ""
        )

        text = self._prepare_text(
            doc.text
        )

        enriched = f"""
        Статья: {article}

        Заголовок:
        {header}

        Текст:
        {text}
        """.strip()

        return (
            query.strip(),
            enriched
        )


    def _prepare_text(
        self,
        text: str
    ) -> str:
        """
        Prepare and truncate text for reranking.

        Args:
            text (str):
                Raw chunk text.

        Returns:
            str:
                Cleaned and truncated text.
        """

        text = (text or "").strip()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        if len(text) <= 1800:
            return text

        head = text[:1200]
        tail = text[-400:]

        return f"{head}\n...\n{tail}"


    def _normalize_logit(
        self,
        score: float
    ) -> float:
        """
        Normalize CrossEncoder logit score.

        Args:
            score (float):
                Raw model output.

        Returns:
            float:
                Normalized semantic relevance score.
        """

        score = float(score)

        return float(
            math.tanh(score / 2.0)
        )


    def _normalize_dense(
        self,
        score: float
    ) -> float:
        """
        Normalize dense retriever similarity score.

        Args:
            score (float):
                Raw dense retriever score.

        Returns:
            float:
                Clamped similarity score in range [0, 1].
        """

        return max(
            0.0,
            min(1.0, float(score))
        )


    def _tokenize(
        self,
        text: str
    ) -> list[str]:
        """
        Tokenize text into normalized words.

        Args:
            text (str):
                Input text.

        Returns:
            List[str]:
                List of normalized tokens.
        """

        return [
            w for w in re.findall(
                r"\w+",
                text.lower()
            )
            if len(w) > 2
        ]


    def _lexical_score(
        self,
        query: str,
        text: str
    ) -> float:
        """
        Compute lexical overlap score.

        Args:
            query (str):
                User query.

            text (str):
                Document text.

        Returns:
            float:
                Lexical similarity score.
        """

        query_words = set(
            self._tokenize(query)
        )

        text_words = set(
            self._tokenize(text[:800])
        )

        if not query_words:
            return 0.0

        overlap = (
            query_words & text_words
        )

        return (
            len(overlap) /
            len(query_words)
        )


    def _header_score(
        self,
        query: str,
        header: str
    ) -> float:
        """
        Compute header relevance boost.

        Args:
            query (str):
                User query.

            header (str):
                Chunk header.

        Returns:
            float:
                Header relevance score.
        """

        q = query.lower().strip()
        h = (header or "").lower().strip()

        if not q or not h:
            return 0.0

        if q == h:
            return self.exact_header_boost

        if q in h:
            return self.partial_header_boost

        q_words = set(
            self._tokenize(q)
        )

        h_words = set(
            self._tokenize(h)
        )

        if not q_words:
            return 0.0

        overlap = (
            q_words & h_words
        )

        ratio = (
            len(overlap) /
            len(q_words)
        )

        return ratio * 0.10


    def _definition_score(
        self,
        query: str,
        text: str,
        header: str
    ) -> float:
        """
        Boost definition-oriented chunks.

        Args:
            query (str):
                User query.

            text (str):
                Chunk text.

            header (str):
                Chunk header.

        Returns:
            float:
                Definition relevance boost.
        """

        query_lower = query.lower()

        triggers = [
            "что такое",
            "понятие",
            "определение",
        ]

        if not any(
            t in query_lower
            for t in triggers
        ):
            return 0.0

        score = 0.0

        text_lower = text.lower()
        header_lower = header.lower()

        legal_patterns = [
            "это",
            "является",
            "признается",
            "понимается",
        ]

        if any(
            p in text_lower[:300]
            for p in legal_patterns
        ):
            score += 0.15

        if "понятие" in header_lower:
            score += 0.10

        return min(
            score,
            self.definition_boost
        )


    def _penalty_score(
        self,
        query: str,
        header: str,
        text: str
    ) -> float:
        """
        Compute heuristic penalties.

        Args:
            query (str):
                User query.

            header (str):
                Chunk header.

            text (str):
                Chunk text.

        Returns:
            float:
                Penalty value.
        """

        penalty = 0.0

        generic_headers = {
            "общие положения",
            "краткое содержание",
            "практическое значение",
        }

        header_lower = (
            header or ""
        ).lower().strip()

        if header_lower in generic_headers:
            penalty += (
                self.generic_header_penalty
            )

        lexical = self._lexical_score(
            query=query,
            text=text
        )

        if lexical < 0.10:
            penalty += (
                self.low_lexical_penalty
            )

        return penalty


    def _diversify(
        self,
        hits: list[SearchResult],
        top_n: int
    ) -> list[SearchResult]:
        """
        Diversify final results by limiting chunks
        from the same article.

        Args:
            hits (List[SearchResult]):
                Ranked search results.

            top_n (int):
                Maximum number of returned results.

        Returns:
            List[SearchResult]:
                Diversified search results.
        """

        selected = []

        article_counts = {}

        for hit in hits:

            article = (
                hit.payload or {}
            ).get(
                "article_number",
                "unknown"
            )

            count = article_counts.get(
                article,
                0
            )

            if count >= self.max_chunks_per_article:
                continue

            selected.append(hit)

            article_counts[article] = (
                count + 1
            )

            if len(selected) >= top_n:
                break

        return selected

    def debug_rerank(
        self,
        query: str,
        hits: list[SearchResult],
        top_n: int = 10
    ) -> None:
        """
        Print detailed reranking debug information.

        Args:
            query (str):
                User query.

            hits (List[SearchResult]):
                Retrieved search results.

            top_n (int):
                Number of debug results to display.

        Returns:
            None
        """

        print("\n" + "=" * 100)

        print("[RERANK DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        ranked = self.rerank(
            query=query,
            hits=hits,
            top_n=top_n
        )

        for idx, hit in enumerate(
            ranked,
            start=1
        ):

            payload = hit.payload or {}

            print(f"\n[{idx}]")

            print(
                f"SCORE   : "
                f"{hit.score:.4f}"
            )

            print(
                f"ARTICLE : "
                f"{payload.get('article_number')}"
            )

            print(
                f"HEADER  : "
                f"{payload.get('header')}"
            )

            print("\nTEXT:")
            print("-" * 80)

            print(
                (hit.text or "")[:1000]
            )

            print("-" * 80)