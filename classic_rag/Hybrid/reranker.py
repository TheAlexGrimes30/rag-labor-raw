from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple
import math
import hashlib

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
        top_n: int = 8,
    ):

        self.model = self._load(model_name)

        self.batch_size = batch_size
        self.max_length = max_length
        self.top_n = top_n

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

            score = self._normalize_score(float(raw_score))

            reranked.append(
                SearchResult.from_rerank(
                    base=hit,
                    score=score
                )
            )

        reranked.sort(
            key=lambda x: x.score,
            reverse=True
        )

        return self._diversify(
            reranked,
            top_n=top_n
        )

    def _build_pair(
        self,
        query: str,
        doc: SearchResult
    ) -> Tuple[str, str]:

        payload = doc.payload or {}

        article = payload.get("article_number", "")
        header = payload.get("header", "")
        topics = payload.get("topics", "")
        source = payload.get("file", "")

        text = self._prepare_text(doc.text)

        enriched_doc = f"""
        Статья: {article}
        
        Раздел: {header}
        
        Темы: {topics}
        
        Источник: {source}
        
        Текст:
        {text}
        """.strip()

        return query, enriched_doc

    def _prepare_text(self, text: str) -> str:

        text = (text or "").strip()

        if len(text) <= 1500:
            return text

        return text[:1500]

    def _normalize_score(self, score: float) -> float:
        """
        sigmoid normalization
        """
        return 1 / (1 + math.exp(-score))

    def _diversify(
        self,
        hits: List[SearchResult],
        top_n: int
    ) -> List[SearchResult]:

        selected = []
        seen = set()

        for h in hits:

            text = (h.text or "").strip()

            key = hashlib.md5(
                text[:300].encode()
            ).hexdigest()

            if key in seen:
                continue

            selected.append(h)
            seen.add(key)

            if len(selected) >= top_n:
                break

        return selected
