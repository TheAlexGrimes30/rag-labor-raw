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


class Reranker(BaseReranker):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    ):
        self._model = self._get_model(model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _get_model(model_name: str) -> CrossEncoder:
        print("Loading Cross-Encoder Reranker...")
        return CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        hits: List["SearchResult"],
        *,
        top_n: int = 6
    ) -> List["SearchResult"]:

        if not hits:
            return []

        filtered_hits = [h for h in hits if h and h.text]

        if not filtered_hits:
            return []

        scores: List[float] = []

        for h in filtered_hits:
            pair = [(str(query), str(h.text).strip())]

            score = self._model.predict(
                pair,
                show_progress_bar=False
            )[0]

            scores.append(float(score))

        ranked = sorted(
            zip(filtered_hits, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            SearchResult.from_rerank(base=h, score=score)
            for h, score in ranked[:top_n]
        ]
