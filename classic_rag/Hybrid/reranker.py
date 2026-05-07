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


class Reranker:

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-Reranker-0.6B",
            batch_size: int = 8,
            max_length: int = 512,
            top_n: int = 6
    ):
        self.model = self._load(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.top_n = top_n

    @staticmethod
    @lru_cache(maxsize=1)
    def _load(model_name: str) -> CrossEncoder:
        model = CrossEncoder(model_name)

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

        valid_hits = [h for h in hits if h.text and h.text.strip()]
        if not valid_hits:
            return []

        pairs = [self._build_pair(query, h) for h in valid_hits]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"[Reranker ERROR] {e}")
            return valid_hits[:top_n]

        scored = [(hit, float(score)) for hit, score in zip(valid_hits, scores)]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        reranked = [
            SearchResult.from_rerank(base=h, score=score)
            for h, score in ranked
        ]

        return self._diversify(reranked, top_n)

    def _build_pair(self, query: str, doc: SearchResult) -> Tuple[str, str]:

        header = doc.payload.get("header") or ""
        article = doc.payload.get("article_number") or ""
        text = self._truncate(doc.text)

        enriched_doc = f"""
        Документ:
        Статья {article}
        Раздел: {header}

        Содержание:
        {text}
        """.strip()

        return query, enriched_doc

    def _truncate(self, text: str) -> str:
        if len(text) <= 1000:
            return text

        return text[:500] + "\n...\n" + text[-500:]

    def _diversify(self, hits: List[SearchResult], top_n: int) -> List[SearchResult]:

        selected = []
        seen = set()

        for h in hits:
            key = (h.payload.get("header"), h.payload.get("article_number"))

            if key in seen:
                continue

            selected.append(h)
            seen.add(key)

            if len(selected) >= top_n:
                break

        return selected