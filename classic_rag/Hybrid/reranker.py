from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

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

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        self.model = self._load(model_name)

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
        top_n: int = 6
    ) -> List[SearchResult]:

        if not hits:
            return []

        hits = [h for h in hits if h.text]

        pairs = [(query, h.text) for h in hits]

        scores = self.model.predict(
            pairs,
            batch_size=1,
            show_progress_bar=False
        )

        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)

        return [
            SearchResult.from_rerank(base=h, score=float(score))
            for h, score in ranked[:top_n]
        ]
