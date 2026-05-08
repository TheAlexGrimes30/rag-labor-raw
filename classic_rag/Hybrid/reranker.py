from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

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
        max_length: int = 1024,
        top_n: int = 5,
        rerank_weight: float = 0.65,
        dense_weight: float = 0.35,
        min_score: float = 0.55,
    ):

        self.model = self._load(model_name)

        self.batch_size = batch_size
        self.max_length = max_length
        self.top_n = top_n

        self.rerank_weight = rerank_weight
        self.dense_weight = dense_weight

        self.min_score = min_score

    @staticmethod
    @lru_cache(maxsize=1)
    def _load(model_name: str) -> CrossEncoder:

        model = CrossEncoder(
            model_name,
            max_length=1024
        )

        tokenizer = model.tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.model.config.pad_token_id = tokenizer.pad_token_id

        return model

    def rerank(
        self,
        query: str,
        hits: List[SearchResult],
        top_n: int | None = None
    ) -> List[SearchResult]:

        if not hits:
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

            dense_score = getattr(hit, "score", 0.0)

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score
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

        article = payload.get("article_number", "")
        header = payload.get("header", "")

        text = self._prepare_text(doc.text)

        enriched_doc = f"""
        Статья {article}

        {header}

        {text}
        """.strip()

        return query.strip(), enriched_doc

    def _prepare_text(self, text: str) -> str:

        text = (text or "").strip()

        if len(text) <= 1200:
            return text

        return text[:1200]

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
                payload.get("article_number"),
                payload.get("header")
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

        for hit, raw_score in zip(valid_hits, raw_scores):

            rerank_score = float(raw_score)

            dense_score = getattr(hit, "score", 0.0)

            final_score = (
                self.rerank_weight * rerank_score +
                self.dense_weight * dense_score
            )

            scored.append(
                (
                    hit,
                    rerank_score,
                    dense_score,
                    final_score
                )
            )

        scored.sort(
            key=lambda x: x[3],
            reverse=True
        )

        for idx, (
            hit,
            rerank_score,
            dense_score,
            final_score
        ) in enumerate(scored[:top_n], start=1):

            payload = hit.payload or {}

            article = payload.get("article_number", "unknown")
            header = payload.get("header", "unknown")

            print(f"\n[{idx}]")

            print(f"RERANK SCORE : {rerank_score:.4f}")
            print(f"DENSE SCORE  : {dense_score:.4f}")
            print(f"FINAL SCORE  : {final_score:.4f}")

            print(f"ARTICLE      : {article}")
            print(f"HEADER       : {header}")

            print("\nTEXT:")
            print("-" * 80)

            print((hit.text or "")[:1000])

            print("-" * 80)
