from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Protocol, Sequence
import hashlib

from classic_rag.Dense.embedder import Embedder
from classic_rag.Dense.search_result import SearchResult

try:
    from llama_index.core.schema import NodeWithScore
except ImportError:  # pragma: no cover
    NodeWithScore = Any  # type: ignore

@dataclass(frozen=True)
class HybridRetrieverConfig:
    """
    Configuration for hybrid dense + Graph RAG retrieval.

    Attributes:
        dense_weight: Weight of dense vector retrieval score in final ranking.
        graph_weight: Weight of Graph RAG retrieval score in final ranking.
        pool_multiplier: Multiplier used to retrieve a larger candidate pool.
        max_pool_size: Maximum number of candidates requested from each retriever.
        min_text_len: Minimum text length allowed in final results.
    """

    dense_weight: float = 0.65
    graph_weight: float = 0.35
    pool_multiplier: int = 8
    max_pool_size: int = 80
    min_text_len: int = 40

class BaseDenseRetriever(ABC):
    """
    Abstract interface for dense vector retrievers.

    Dense retrievers perform semantic similarity search using embedding vectors.
    """

    @abstractmethod
    def search(self, query_vec: list[float], k: int) -> list[SearchResult]:
        """
        Execute dense vector similarity search.

        Args:
            query_vec: Query embedding vector.
            k: Number of documents to retrieve.

        Returns:
            Retrieved search results.
        """

        raise NotImplementedError


class BaseGraphRetriever(ABC):
    """
    Abstract interface for Graph RAG retrievers.

    Graph retrievers search over an entity-relation graph and return text nodes
    connected to the entities or paths relevant to the user query.
    """

    @abstractmethod
    def search(self, query: str, k: int) -> list[SearchResult]:
        """
        Execute graph-based retrieval.

        Args:
            query: User query text.
            k: Number of graph candidates to retrieve.

        Returns:
            Retrieved graph search results.
        """

        raise NotImplementedError


class BaseRetriever(ABC):
    """
    Abstract interface for application retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query.
            top_k: Number of chunks to return.

        Returns:
            Retrieved chunks.
        """

        raise NotImplementedError


class LlamaIndexRetrieverProtocol(Protocol):
    """
    Protocol for LlamaIndex retrievers.

    PropertyGraphIndex.as_retriever(...) returns an object with retrieve(query),
    therefore the hybrid retriever depends on behavior rather than concrete class.
    """

    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        """
        Retrieve LlamaIndex nodes for a query.
        """

        ...

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