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
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
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

        filtered_hits: List["SearchResult"] = [
            h for h in hits if h is not None
        ]

        if not filtered_hits:
            return []

        pairs: List[Tuple[str, str]] = [
            (query, h.text.strip())
            for h in filtered_hits
        ]

        scores = self._predict_batched(pairs)

        ranked = sorted(
            zip(filtered_hits, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            SearchResult.from_rerank(base=h, score=float(score))
            for h, score in ranked[:top_n]
        ]

    def _predict_batched(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:

        try:
            scores = self._model.predict(
                pairs,
                show_progress_bar=False
            )

        except ValueError as e:
            if "no padding token" in str(e):
                print("Model does not support batching → fallback to batch_size=1")

                scores = []
                for q, doc in pairs:
                    s = self._model.predict(
                        [(q, doc)],
                        show_progress_bar=False
                    )[0]
                    scores.append(float(s))

                return scores

            raise

        if not isinstance(scores, list):
            scores = scores.tolist()

        return [float(s) for s in scores]