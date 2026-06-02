from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, Protocol, Sequence, Any, Dict, Optional
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
    """
    Dense retriever implementation based on Qdrant vector search.
    """

    def __init__(self, vector_store: Any) -> None:
        """
        Initialize Qdrant dense retriever.

        Args:
            vector_store: Qdrant wrapper with search(query_vector, limit).
        """

        self.vector_store = vector_store

    def search(self, query_vec: list[float], k: int) -> list[SearchResult]:
        """
        Search Qdrant by query vector.
        """

        hits = self.vector_store.search(query_vector=query_vec, limit=k)
        results: list[SearchResult] = []

        for hit in hits:
            search_result = SearchResult.from_qdrant(hit)

            if search_result.text and search_result.text.strip():
                results.append(search_result)

        return results


class LlamaIndexGraphRetriever(BaseGraphRetriever):
    """
    Graph RAG retriever adapter for LlamaIndex.

    The class accepts a prepared LlamaIndex retriever, for example a retriever
    created from PropertyGraphIndex.as_retriever(...). This keeps graph building
    outside retrieval and makes the class easy to test.
    """

    def __init__(self, graph_retriever: LlamaIndexRetrieverProtocol) -> None:
        """
        Initialize graph retriever adapter.

        Args:
            graph_retriever: LlamaIndex graph retriever instance.
        """

        self.graph_retriever = graph_retriever

    def search(self, query: str, k: int) -> list[SearchResult]:
        """
        Search the graph and convert LlamaIndex nodes to SearchResult objects.
        """

        nodes = list(self.graph_retriever.retrieve(query))[:k]
        return [self._node_to_search_result(node) for node in nodes]

    def _node_to_search_result(self, node: NodeWithScore) -> SearchResult:
        """
        Convert a LlamaIndex NodeWithScore into the project's SearchResult.
        """

        raw_node = getattr(node, "node", node)
        score = float(getattr(node, "score", 0.0) or 0.0)

        node_id = (
            getattr(raw_node, "node_id", None)
            or getattr(raw_node, "id_", None)
            or getattr(raw_node, "id", None)
        )

        text = ""

        if hasattr(raw_node, "get_content"):
            text = raw_node.get_content() or ""
        else:
            text = getattr(raw_node, "text", "") or ""

        metadata = getattr(raw_node, "metadata", None) or {}
        payload = dict(metadata)
        payload["retrieval_source"] = "graph"

        return self._make_search_result(
            id=str(node_id) if node_id else None,
            text=text,
            score=score,
            payload=payload,
        )

    @staticmethod
    def _make_search_result(
        *,
        id: Optional[str],
        text: str,
        score: float,
        payload: Dict[str, Any],
    ) -> SearchResult:
        """
        Create SearchResult without depending on a specific constructor shape.
        """

        try:
            return SearchResult(id=id, text=text, score=score, payload=payload)
        except TypeError:
            result = SearchResult.__new__(SearchResult)
            result.id = id
            result.text = text
            result.score = score
            result.payload = payload
            return result


class HybridFusionService:
    """
    Service responsible only for merging, normalizing and deduplicating results.
    """

    def __init__(self, config: HybridRetrieverConfig) -> None:
        """
        Initialize fusion service.

        Args:
            config: Hybrid retrieval configuration.
        """

        self.config = config

    def fuse(
        self,
        *,
        dense_results: list[SearchResult],
        graph_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Fuse dense and graph results into one ranked list.

        Args:
            dense_results: Candidates from vector search.
            graph_results: Candidates from Graph RAG search.
            top_k: Required number of final results.

        Returns:
            Deduplicated and reranked results.
        """

        normalized_dense = self._normalize_scores(dense_results)
        normalized_graph = self._normalize_scores(graph_results)
        merged: dict[str, SearchResult] = {}
        source_scores: dict[str, float] = {}

        self._add_results(
            merged=merged,
            source_scores=source_scores,
            results=normalized_dense,
            weight=self.config.dense_weight,
            source="dense",
        )

        self._add_results(
            merged=merged,
            source_scores=source_scores,
            results=normalized_graph,
            weight=self.config.graph_weight,
            source="graph",
        )

        fused_results = list(merged.values())

        for result in fused_results:
            key = self._dedupe_key(result)
            result.score = source_scores[key]

        filtered = self._basic_filter(fused_results)
        filtered.sort(key=lambda item: item.score, reverse=True)

        return filtered[:top_k]

    def _add_results(
        self,
        *,
        merged: dict[str, SearchResult],
        source_scores: dict[str, float],
        results: list[SearchResult],
        weight: float,
        source: str,
    ) -> None:
        """
        Add weighted results from one retrieval source to the merged pool.
        """

        for result in results:
            key = self._dedupe_key(result)
            weighted_score = float(result.score or 0.0) * weight

            if key not in merged:
                merged[key] = result
                source_scores[key] = weighted_score
            else:
                source_scores[key] += weighted_score
                merged[key].payload = self._merge_payloads(
                    merged[key].payload or {},
                    result.payload or {},
                    source,
                )

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Normalize scores into the 0..1 range for fair dense/graph fusion.
        """

        if not results:
            return []

        scores = [float(result.score or 0.0) for result in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for result in results:
                result.score = 1.0
            return results

        for result in results:
            result.score = (float(result.score or 0.0) - min_score) / (max_score - min_score)

        return results

    def _basic_filter(self, hits: list[SearchResult]) -> list[SearchResult]:
        """
        Remove empty, too short and duplicate chunks.
        """

        seen: Set[str] = set()
        result: list[SearchResult] = []

        for hit in hits:
            text = (hit.text or "").strip()

            if len(text) < self.config.min_text_len:
                continue

            key = self._dedupe_key(hit)

            if key in seen:
                continue

            seen.add(key)
            result.append(hit)

        return result

    @staticmethod
    def _dedupe_key(result: SearchResult) -> str:
        """
        Build stable deduplication key for a result.
        """

        if getattr(result, "id", None):
            return str(result.id)

        text = (getattr(result, "text", "") or "")[:500]
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _merge_payloads(
        left: dict[str, Any],
        right: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        """
        Merge metadata from duplicate dense and graph results.
        """

        merged = dict(left)
        merged.update({key: value for key, value in right.items() if key not in merged})

        sources = set(merged.get("retrieval_sources", []))
        sources.add(source)
        merged["retrieval_sources"] = sorted(sources)

        return merged


class HybridGraphRetriever(BaseRetriever):
    """
    Application retriever using hybrid Dense RAG + Graph RAG retrieval.

    Flow:
        1. Convert query to embedding.
        2. Retrieve dense candidates from Qdrant.
        3. Retrieve graph candidates from LlamaIndex Graph RAG.
        4. Normalize scores.
        5. Fuse, deduplicate and return final top_k chunks.
    """

    def __init__(
        self,
        vector_store: Any,
        embedder: Embedder,
        graph_retriever: BaseGraphRetriever,
        *,
        config: Optional[HybridRetrieverConfig] = None,
    ) -> None:
        """
        Initialize hybrid graph retriever.

        Args:
            vector_store: Qdrant vector store wrapper.
            embedder: Query embedder.
            graph_retriever: Graph RAG retriever implementation.
            config: Optional hybrid retrieval configuration.
        """

        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or HybridRetrieverConfig()

        self.dense = QdrantDenseRetriever(vector_store)
        self.graph = graph_retriever
        self.fusion = HybridFusionService(self.config)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrieve relevant chunks using dense and graph retrieval together.
        """

        query = (query or "").strip()

        if not query:
            return []

        pool_size = min(
            self.config.max_pool_size,
            max(top_k * self.config.pool_multiplier, 30),
        )

        query_vec = self.embedder.encode_queries([query])[0]

        dense_candidates = self.dense.search(query_vec=query_vec, k=pool_size)
        graph_candidates = self.graph.search(query=query, k=pool_size)

        return self.fusion.fuse(
            dense_results=dense_candidates,
            graph_results=graph_candidates,
            top_k=top_k,
        )

    def debug_query(self, query: str, top_k: int = 10) -> None:
        """
        Print debug information for hybrid retrieval.
        """

        print("\n" + "=" * 100)
        print("[HYBRID GRAPH RETRIEVAL DEBUG]")
        print(f"QUERY: {query}")
        print("=" * 100)

        hits = self.retrieve(query=query, top_k=top_k)

        if not hits:
            print("No hits")
            return

        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            article = payload.get("article_number", "unknown")
            header = payload.get("header", "unknown")
            source = payload.get("retrieval_source") or payload.get("retrieval_sources", "unknown")

            print("\n" + "-" * 100)
            print(f"TOP {index}")
            print(f"SCORE   : {hit.score:.4f}")
            print(f"SOURCE  : {source}")
            print(f"ARTICLE : {article}")
            print(f"HEADER  : {header}")
            print(f"ID      : {hit.id}")
            print("\nTEXT:\n")
            print((hit.text or "")[:1200])


Retriever = HybridGraphRetriever
