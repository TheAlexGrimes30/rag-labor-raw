from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

import math
import re

from sentence_transformers import CrossEncoder

from classic_rag.Hybrid.rag_config import SearchResult


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: List["SearchResult"],
        *,
        top_n: int
    ) -> List["SearchResult"]:
        raise NotImplementedError


class Reranker(BaseReranker):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        batch_size: int = 8,
        max_length: int = 512,
        top_n: int = 5,

        rerank_weight: float = 0.85,
        dense_weight: float = 0.10,
        lexical_weight: float = 0.05,

        min_score: float = 0.45,
        relative_threshold: float = 0.75,

        exact_header_boost: float = 0.20,
        partial_header_boost: float = 0.10,
        generic_header_penalty: float = 0.04,
        low_lexical_penalty: float = 0.03,

        max_chunks_per_article: int = 2,
    ):

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

        self.min_score = min_score
        self.relative_threshold = relative_threshold

        self.exact_header_boost = exact_header_boost
        self.partial_header_boost = partial_header_boost

        self.generic_header_penalty = generic_header_penalty
        self.low_lexical_penalty = low_lexical_penalty

        self.max_chunks_per_article = max_chunks_per_article


    @staticmethod
    @lru_cache(maxsize=1)
    def _load(
        model_name: str,
        max_length: int
    ) -> CrossEncoder:

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
        hits: List[SearchResult],
        top_n: int | None = None
    ) -> List[SearchResult]:

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
            self._build_pair(query, hit)
            for hit in valid_hits
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

        reranked = []

        for hit, raw_score in zip(valid_hits, raw_scores):

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

            penalty = self._penalty_score(
                query=query,
                header=header,
                text=hit.text or ""
            )

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                header_score -
                penalty
            )

            reranked.append(
                SearchResult.from_rerank(
                    base=hit,
                    score=final_score
                )
            )

        reranked.sort(
            key=lambda x: x.score,
            reverse=True
        )

        reranked = self._dynamic_filter(
            reranked
        )

        reranked = self._diversify(
            reranked,
            top_n=top_n
        )

        return reranked[:top_n]

    def _build_pair(
        self,
        query: str,
        doc: SearchResult
    ) -> Tuple[str, str]:

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

        enriched_doc = f"""
        Статья: {article}
        
        Заголовок:
        {header}
        
        Текст:
        {text}
        """.strip()

        return (
            query.strip(),
            enriched_doc
        )

    def _prepare_text(
        self,
        text: str
    ) -> str:

        text = (text or "").strip()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        if len(text) <= 1200:
            return text

        head = text[:800]
        tail = text[-300:]

        return (
            f"{head}\n...\n{tail}"
        )

    def _normalize_logit(
        self,
        score: float
    ) -> float:

        score = float(score)

        return 1 / (
            1 + math.exp(-score)
        )

    def _normalize_dense(
        self,
        score: float
    ) -> float:

        score = float(score)

        return max(
            0.0,
            min(1.0, score)
        )

    def _tokenize(
        self,
        text: str
    ) -> List[str]:

        words = re.findall(
            r"\w+",
            text.lower()
        )

        return [
            w for w in words
            if len(w) > 2
        ]

    def _lexical_score(
        self,
        query: str,
        text: str
    ) -> float:


        text = text[:300]

        query_words = set(
            self._tokenize(query)
        )

        text_words = set(
            self._tokenize(text)
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

        q = query.lower().strip()
        h = (header or "").lower().strip()

        if not q or not h:
            return 0.0

        # exact match
        if q == h:
            return self.exact_header_boost

        if q in h:
            return self.partial_header_boost

        query_words = set(
            self._tokenize(q)
        )

        header_words = set(
            self._tokenize(h)
        )

        if not query_words:
            return 0.0

        overlap = (
            query_words & header_words
        )

        ratio = (
            len(overlap) /
            len(query_words)
        )

        return ratio * 0.08

    def _penalty_score(
        self,
        query: str,
        header: str,
        text: str
    ) -> float:

        penalty = 0.0

        header_lower = (
            header or ""
        ).lower().strip()

        generic_headers = {
            "общие положения",
            "понятие",
            "основные положения",
            "краткое содержание",
            "практическое значение",
        }

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

    def _dynamic_filter(
        self,
        hits: List[SearchResult]
    ) -> List[SearchResult]:

        if not hits:
            return []

        best_score = hits[0].score

        dynamic_threshold = max(
            self.min_score,
            best_score * self.relative_threshold
        )

        filtered = [
            h for h in hits
            if h.score >= dynamic_threshold
        ]

        if not filtered:
            return hits[:self.top_n]

        return filtered

    def _diversify(
        self,
        hits: List[SearchResult],
        top_n: int
    ) -> List[SearchResult]:

        selected = []

        article_counts = {}

        for hit in hits:

            payload = hit.payload or {}

            article = payload.get(
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

            article_counts[article] = count + 1

            if len(selected) >= top_n:
                break

        return selected

    def debug_rerank(
        self,
        query: str,
        hits: List[SearchResult],
        top_n: int = 10
    ):

        print("\n" + "=" * 100)

        print("[RERANK DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        if not hits:
            print("No hits")
            return

        valid_hits = [
            h for h in hits
            if h.text and h.text.strip()
        ]

        if not valid_hits:
            print("No valid hits")
            return

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

            print(f"[DEBUG ERROR] {e}")
            return

        scored = []

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

            penalty = self._penalty_score(
                query=query,
                header=header,
                text=hit.text or ""
            )

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                header_score -
                penalty
            )

            scored.append(
                (
                    hit,
                    rerank_score,
                    dense_score,
                    lexical_score,
                    header_score,
                    penalty,
                    final_score
                )
            )

        scored.sort(
            key=lambda x: x[6],
            reverse=True
        )

        for idx, (
            hit,
            rerank_score,
            dense_score,
            lexical_score,
            header_score,
            penalty,
            final_score
        ) in enumerate(scored[:top_n], start=1):

            payload = hit.payload or {}

            article = payload.get(
                "article_number",
                "unknown"
            )

            header = payload.get(
                "header",
                "unknown"
            )

            print(f"\n[{idx}]")

            print(
                f"RERANK SCORE : "
                f"{rerank_score:.4f}"
            )

            print(
                f"DENSE SCORE  : "
                f"{dense_score:.4f}"
            )

            print(
                f"LEXICAL      : "
                f"{lexical_score:.4f}"
            )

            print(
                f"HEADER SCORE : "
                f"{header_score:.4f}"
            )

            print(
                f"PENALTY      : "
                f"{penalty:.4f}"
            )

            print(
                f"FINAL SCORE  : "
                f"{final_score:.4f}"
            )

            print(
                f"ARTICLE      : "
                f"{article}"
            )

            print(
                f"HEADER       : "
                f"{header}"
            )

            print("\nTEXT:")

            print("-" * 80)

            print(
                (hit.text or "")[:1200]
            )

            print("-" * 80)