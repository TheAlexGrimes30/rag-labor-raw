from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Set

from sentence_transformers import SentenceTransformer

from classic_rag.Hybrid.rag_config import (
    SearchResult,
    Chunk
)
from classic_rag.Hybrid.storage import VectorStore



class BaseDenseRetriever(ABC):

    @abstractmethod
    def search(
        self,
        query_vec: List[float],
        k: int
    ) -> List[SearchResult]:
        raise NotImplementedError


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        raise NotImplementedError



@dataclass
class Embedder:

    model_name: str
    batch_size: int = 16
    normalize: bool = True

    def __post_init__(self):

        self._model = self._load_model(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        return SentenceTransformer(model_name)

    def encode_queries(self, texts: List[str]) -> List[List[float]]:
        return self._encode(self._apply_prefix(texts, True))

    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        return self._encode(self._apply_prefix(texts, False))

    def _apply_prefix(self, texts: List[str], is_query: bool) -> List[str]:

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

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(
        self,
        query_vec: List[float],
        k: int
    ) -> List[SearchResult]:

        hits = self.vector_store.search(
            query_vector=query_vec,
            limit=k
        )

        results = []

        for hit in hits:
            sr = SearchResult.from_qdrant(hit)

            if sr.text and sr.text.strip():
                results.append(sr)

        return results



class Retriever(BaseRetriever):

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.dense = QdrantDenseRetriever(vector_store)


    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:

        query = query.strip()

        query_vec = self.embedder.encode_queries([query])[0]

        # oversampling for dedup
        results = self.dense.search(
            query_vec=query_vec,
            k=top_k * 3
        )

        return self._deduplicate(results)[:top_k]


    def _deduplicate(
        self,
        hits: List[SearchResult]
    ) -> List[SearchResult]:

        seen: Set[str] = set()
        result: List[SearchResult] = []

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < 50:
                continue

            # stable key
            key = h.id or text[:200]

            if key in seen:
                continue

            seen.add(key)
            result.append(h)

        return result


    def debug_query(
        self,
        query: str,
        top_k: int = 10
    ) -> None:

        print("\n" + "=" * 80)
        print(f"[QUERY] {query}")

        query_vec = self.embedder.encode_queries([query])[0]

        hits = self.dense.search(
            query_vec=query_vec,
            k=top_k
        )

        print(f"\n[DENSE TOP {top_k}]")

        for i, h in enumerate(hits):

            print(
                f"{i+1}. score={h.score:.4f} | id={h.id}"
            )

            print(h.text[:400])
            print()

    def debug_embedding_inputs(
        self,
        chunks: List[Chunk]
    ) -> None:

        print("\n" + "=" * 80)
        print("[EMBEDDING INPUT DEBUG]")
        print("=" * 80)

        for i, chunk in enumerate(chunks[:5]):

            print(f"\nCHUNK {i}")
            print("-" * 80)

            print("\nRAW TEXT:")
            print(chunk.text[:500])

            print("\nEMBEDDING TEXT (IMPORTANT):")
            print(self._build_embedding_text(chunk)[:1000])

    def _build_embedding_text(self, chunk: Chunk) -> str:

        parts = []

        if chunk.metadata.source:
            parts.append(f"Документ: {chunk.metadata.source}")

        if chunk.metadata.article_number:
            parts.append(f"Статья {chunk.metadata.article_number}")

        if chunk.metadata.header:
            parts.append(f"Раздел: {chunk.metadata.header}")

        if chunk.metadata.topics:
            topics = ", ".join(
                t.replace("_", " ")
                for t in chunk.metadata.topics
            )
            parts.append(f"Темы: {topics}")

        parts.append(chunk.text)

        return "\n".join(parts)

    def debug_query_embedding(self, query: str) -> None:
        """
        Проверка embedding запроса:
        - первые значения вектора
        - L2-норма (должна быть ~1 если normalize=True)
        """

        query = query.strip()

        query_vec = self.embedder.encode_queries([query])[0]

        print("\n" + "=" * 80)
        print("[QUERY EMBEDDING DEBUG]")
        print("=" * 80)

        print("QUERY:")
        print(query)

        print("\nVECTOR SAMPLE (first 10 dims):")
        print(query_vec[:10])

        norm = sum(x * x for x in query_vec) ** 0.5

        print("\nL2 NORM:")
        print(norm)

        print("=" * 80)
