from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Set
import hashlib

from sentence_transformers import SentenceTransformer
from classic_rag.Hybrid.rag_config import SearchResult

class BaseDenseRetriever(ABC):

    @abstractmethod
    def search(self, query_vec: List[float], k: int) -> List[SearchResult]:
        raise NotImplementedError


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        raise NotImplementedError


class Embedder:

    def __init__(self, model_name: str, batch_size: int = 16, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self._model = self._load_model(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str):
        return SentenceTransformer(model_name)

    def encode_queries(self, texts: List[str]) -> List[List[float]]:
        texts = self._apply_prefix(texts, is_query=True)
        return self._encode(texts)

    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        texts = self._apply_prefix(texts, is_query=False)
        return self._encode(texts)

    def _apply_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        # only for E5-like models
        if "e5" not in self.model_name.lower():
            return texts

        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]

    def _encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return vectors.tolist()


class QdrantDenseRetriever(BaseDenseRetriever):

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query_vec: List[float], k: int) -> List[SearchResult]:
        hits = self.vector_store.search(query_vector=query_vec, limit=k)

        results = []
        for hit in hits:
            sr = SearchResult.from_qdrant(hit)

            if sr.text and sr.text.strip():
                results.append(sr)

        return results


class Retriever(BaseRetriever):

    def __init__(
        self,
        vector_store,
        embedder: Embedder,
        *,
        pool_multiplier: int = 8,
        max_pool_size: int = 80,
        min_text_len: int = 40
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.dense = QdrantDenseRetriever(vector_store)

        self.pool_multiplier = pool_multiplier
        self.max_pool_size = max_pool_size
        self.min_text_len = min_text_len


    def _expand_query(self, query: str) -> List[str]:
        """
        Legal-domain safe expansion (NO semantic loss).
        """
        return [
            query,
            query.replace("что такое", "").strip(),
            query.replace("понятие", "").strip(),
            query
        ]

    def _embed_query(self, query: str) -> List[float]:
        variants = self._expand_query(query)

        vectors = self.embedder.encode_queries(variants)

        # mean pooling (VERY important for recall)
        dim = len(vectors[0])
        avg = [0.0] * dim

        for v in vectors:
            for i in range(dim):
                avg[i] += v[i]

        return [x / len(vectors) for x in avg]


    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:

        query_vec = self._embed_query(query)

        pool_size = min(
            self.max_pool_size,
            max(top_k * self.pool_multiplier, 30)
        )

        candidates = self.dense.search(query_vec=query_vec, k=pool_size)

        candidates = self._basic_filter(candidates)

        return candidates

    def _basic_filter(self, hits: List[SearchResult]) -> List[SearchResult]:

        seen: Set[str] = set()
        result: List[SearchResult] = []

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < self.min_text_len:
                continue

            key = h.id or hashlib.md5(text[:200].encode()).hexdigest()

            if key in seen:
                continue

            seen.add(key)
            result.append(h)

        return result


    def debug_query(self, query: str, top_k: int = 10):

        print("\n" + "=" * 80)
        print(f"[QUERY] {query}")

        query_vec = self._embed_query(query)

        hits = self.dense.search(query_vec=query_vec, k=top_k)

        print(f"\n[DENSE TOP {top_k}]")

        for i, h in enumerate(hits):
            print(f"{i+1}. score={h.score:.4f} | id={h.id}")
            print(h.text[:400])
            print()