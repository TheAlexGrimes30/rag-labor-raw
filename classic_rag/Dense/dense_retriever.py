from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Set
import hashlib

from sentence_transformers import SentenceTransformer

from classic_rag.Dense.search_result import SearchResult


class BaseDenseRetriever(ABC):
    """
    Abstract interface for dense vector retrievers.

    Dense retrievers perform semantic similarity search
    using embedding vectors.
    """

    @abstractmethod
    def search(
        self,
        query_vec: list[float],
        k: int
    ) -> list[SearchResult]:
        """
        Execute dense vector similarity search.

        Args:
            query_vec (List[float]):
                Query embedding vector.

            k (int):
                Number of documents to retrieve.

        Returns:
            List[SearchResult]:
                Retrieved search results.
        """

        raise NotImplementedError


class BaseRetriever(ABC):
    """
    Abstract interface for retrievers.

    Retriever converts text query into embeddings
    and returns relevant search results.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query (str):
                User query.

            top_k (int):
                Number of chunks to return.

        Returns:
            List[SearchResult]:
                Retrieved chunks.
        """

        raise NotImplementedError


class Embedder:
    """
    Wrapper around SentenceTransformer models.

    Responsible for:
    - loading embedding model
    - query/document encoding
    - E5 prefix handling
    - vector normalization
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self._model = self._load_model(model_name)

        self.dim = (
            self._model.get_sentence_embedding_dimension()
        )

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        """
        Load and cache embedding model.

        Args:
            model_name (str):
                HuggingFace model name.

        Returns:
            SentenceTransformer:
                Loaded embedding model.
        """

        return SentenceTransformer(model_name)

    def encode_queries(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode search queries.

        Args:
            texts (List[str]):
                Query texts.

        Returns:
            List[List[float]]:
                Query embeddings.
        """


        texts = self._apply_prefix(
            texts,
            is_query=True
        )

        return self._encode(texts)

    def encode_passages(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode document passages.

        Args:
            texts (List[str]):
                Passage texts.

        Returns:
            List[List[float]]:
                Passage embeddings.
        """

        texts = self._apply_prefix(
            texts,
            is_query=False
        )

        return self._encode(texts)

    def _apply_prefix(
        self,
        texts: List[str],
        is_query: bool
    ) -> List[str]:
        """
        Apply E5 prefixes if model requires them.

        E5 models require:
        - "query: " for queries
        - "passage: " for documents

        Args:
            texts (List[str]):
                Input texts.

            is_query (bool):
                Whether texts are queries.

         Returns:
            List[str]:
                Prefixed texts.
        """

        if "e5" not in self.model_name.lower():
            return texts

        prefix = (
            "query: "
            if is_query
            else "passage: "
        )

        return [
            prefix + t
            for t in texts
        ]

    def _encode(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts (List[str]):
                Input texts.

        Returns:
            List[List[float]]:
                Embedding vectors.
        """

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
        vector_store,
        embedder: Embedder,
        *,
        pool_multiplier: int = 8,
        max_pool_size: int = 80,
        min_text_len: int = 40
    ):

        self.vector_store = vector_store
        self.embedder = embedder

        self.dense = QdrantDenseRetriever(
            vector_store
        )

        self.pool_multiplier = pool_multiplier
        self.max_pool_size = max_pool_size
        self.min_text_len = min_text_len

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:

        query = (query or "").strip()

        if not query:
            return []

        query_vec = self.embedder.encode_queries(
            [query]
        )[0]

        pool_size = min(
            self.max_pool_size,
            max(top_k * self.pool_multiplier, 30)
        )

        candidates = self.dense.search(
            query_vec=query_vec,
            k=pool_size
        )

        candidates = self._basic_filter(
            candidates
        )

        return candidates[:top_k]

    def _basic_filter(
        self,
        hits: List[SearchResult]
    ) -> List[SearchResult]:

        seen: Set[str] = set()

        result: List[SearchResult] = []

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < self.min_text_len:
                continue

            key = (
                h.id
                or hashlib.md5(
                    text[:200].encode()
                ).hexdigest()
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(h)

        return result

    def debug_query(
            self,
            query: str,
            top_k: int = 10
    ):

        print("\n" + "=" * 100)

        print(f"[DENSE RETRIEVAL DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        query = (query or "").strip()

        if not query:
            print("Empty query")

            return

        query_vec = self.embedder.encode_queries(
            [query]
        )[0]

        hits = self.dense.search(
            query_vec=query_vec,
            k=top_k
        )

        hits = self._basic_filter(hits)

        if not hits:
            print("No hits")

            return

        for i, h in enumerate(
                hits,
                start=1
        ):
            payload = h.payload or {}

            article = payload.get(
                "article_number",
                "unknown"
            )

            header = payload.get(
                "header",
                "unknown"
            )

            print("\n" + "-" * 100)

            print(f"TOP {i}")

            print(
                f"SCORE   : "
                f"{h.score:.4f}"
            )

            print(
                f"ARTICLE : "
                f"{article}"
            )

            print(
                f"HEADER  : "
                f"{header}"
            )

            print(
                f"ID      : "
                f"{h.id}"
            )

            print("\nTEXT:\n")

            print(
                (h.text or "")[:1200]
            )