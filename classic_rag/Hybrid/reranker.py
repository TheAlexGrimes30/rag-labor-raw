from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple
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
        rerank_weight: float = 0.45,
        dense_weight: float = 0.45,
        lexical_weight: float = 0.10,
        min_score: float = 0.68,
    ):

        self.model = self._load(
            model_name,
            max_length=max_length
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.top_n = top_n

        self.rerank_weight = rerank_weight
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        self.min_score = min_score

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
            self._build_pair(query, h)
            for h in valid_hits
        ]

        try:

            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

        except Exception as e:

            print(f"[RERANK ERROR] {e}")

            return sorted(
                valid_hits,
                key=lambda x: getattr(x, "score", 0),
                reverse=True
            )[:top_n]

        reranked = []

        for hit, raw_score in zip(valid_hits, scores):

            rerank_score = float(raw_score)

            rerank_score = max(
                0.0,
                min(1.0, rerank_score)
            )

            dense_score = float(
                getattr(hit, "score", 0.0)
            )

            dense_score = max(
                0.0,
                min(1.0, dense_score)
            )

            lexical_score = self._lexical_overlap(
                query=query,
                text=hit.text or ""
            )

            payload = hit.payload or {}

            header_boost = self._header_boost(
                query=query,
                header=payload.get("header", "")
            )

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                header_boost
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

        reranked = [
            h for h in reranked
            if h.score >= self.min_score
        ]

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
Статья {article}

{header}

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

        if len(text) <= 1200:
            return text

        return text[:1200]

    def _lexical_overlap(
        self,
        query: str,
        text: str
    ) -> float:

        query_words = set(
            re.findall(
                r"\w+",
                query.lower()
            )
        )

        text_words = set(
            re.findall(
                r"\w+",
                text.lower()
            )
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

    def _header_boost(
        self,
        query: str,
        header: str
    ) -> float:

        query = query.lower().strip()

        header = (
            header or ""
        ).lower().strip()

        if not header:
            return 0.0

        if query in header:
            return 0.15

        query_words = query.split()

        matched = sum(
            1
            for w in query_words
            if w in header
        )

        if matched >= 2:
            return 0.10

        if matched >= 1:
            return 0.05

        return 0.0

    def _diversify(
        self,
        hits: List[SearchResult],
        top_n: int
    ) -> List[SearchResult]:

        selected = []

        seen = set()

        for h in hits:

            payload = h.payload or {}

            key = (
                payload.get(
                    "article_number"
                ),
                payload.get(
                    "header"
                )
            )

            if key in seen:
                continue

            selected.append(h)

            seen.add(key)

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

            rerank_score = float(raw_score)

            rerank_score = max(
                0.0,
                min(1.0, rerank_score)
            )

            dense_score = float(
                getattr(hit, "score", 0.0)
            )

            dense_score = max(
                0.0,
                min(1.0, dense_score)
            )

            lexical_score = self._lexical_overlap(
                query=query,
                text=hit.text or ""
            )

            payload = hit.payload or {}

            header_boost = self._header_boost(
                query=query,
                header=payload.get("header", "")
            )

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                header_boost
            )

            scored.append(
                (
                    hit,
                    rerank_score,
                    dense_score,
                    lexical_score,
                    header_boost,
                    final_score
                )
            )

        scored.sort(
            key=lambda x: x[5],
            reverse=True
        )

        for idx, (
            hit,
            rerank_score,
            dense_score,
            lexical_score,
            header_boost,
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
                f"HEADER BOOST : "
                f"{header_boost:.4f}"
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
                (hit.text or "")[:1000]
            )

            print("-" * 80)