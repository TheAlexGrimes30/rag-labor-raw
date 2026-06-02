from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Set
import hashlib
import math
import re

from rank_bm25 import BM25Okapi
from llama_index.core import Document

from classic_rag.Dense.embedder import Embedder
from classic_rag.Dense.search_result import SearchResult


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """
    Configuration for Dense + BM25 + GraphRAG hybrid retrieval.
    """

    alpha: float = 0.8
    graph_weight: float = 0.1
    pool_multiplier: int = 8
    max_pool_size: int = 80
    min_text_len: int = 40


class BaseDenseRetriever(ABC):
    """
    Abstract interface for dense retrievers.
    """

    @abstractmethod
    def search(
            self,
            query_vec: list[float],
            k: int
    ) -> list[SearchResult]:
        """
        Search by dense vector.
        """

        raise NotImplementedError


class BaseSparseRetriever(ABC):
    """
    Abstract interface for sparse retrievers.
    """

    @abstractmethod
    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search by text query.
        """

        raise NotImplementedError


class BaseGraphRetriever(ABC):
    """
    Abstract interface for GraphRAG retrievers.
    """

    @abstractmethod
    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search by graph context.
        """

        raise NotImplementedError


class BaseRetriever(ABC):
    """
    Abstract application retriever.
    """

    @abstractmethod
    def retrieve(
            self,
            query: str,
            top_k: int = 10
    ) -> list[SearchResult]:
        """
        Retrieve relevant chunks.
        """

        raise NotImplementedError


class SearchResultFactory:
    """
    Factory for SearchResult objects.
    """

    @staticmethod
    def create(
            *,
            id: Optional[str],
            text: str,
            score: float,
            payload: dict[str, Any]
    ) -> SearchResult:
        """
        Create SearchResult regardless of constructor shape.
        """

        try:
            return SearchResult(
                id=id,
                text=text,
                score=score,
                payload=payload
            )
        except TypeError:
            result = SearchResult.__new__(SearchResult)
            result.id = id
            result.text = text
            result.score = score
            result.payload = payload
            return result


class ChunkAdapter:
    """
    Converts project chunks into plain fields and LlamaIndex documents.
    """

    @staticmethod
    def metadata_to_dict(
            metadata: Any
    ) -> dict[str, Any]:
        """
        Convert metadata object to dict.
        """

        if metadata is None:
            return {}

        if isinstance(metadata, dict):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(metadata.model_dump())

        if hasattr(metadata, "dict"):
            return dict(metadata.dict())

        if hasattr(metadata, "__dict__"):
            return {
                key: value
                for key, value in metadata.__dict__.items()
                if not key.startswith("_")
            }

        return {}

    @staticmethod
    def extract_text(
            chunk: Any
    ) -> str:
        """
        Extract text from chunk.
        """

        text = (
            getattr(chunk, "text", None)
            or getattr(chunk, "content", None)
            or getattr(chunk, "page_content", None)
            or ""
        )

        return str(text)

    @classmethod
    def extract_metadata(
            cls,
            chunk: Any
    ) -> dict[str, Any]:
        """
        Extract normalized metadata from chunk.
        """

        metadata_raw = (
            getattr(chunk, "metadata", None)
            or getattr(chunk, "payload", None)
            or {}
        )

        metadata = cls.metadata_to_dict(metadata_raw)

        article_number = (
            metadata.get("article_number")
            or metadata.get("article")
        )

        if article_number is not None:
            metadata["article_number"] = str(article_number).strip()

        header = metadata.get("header")

        if header is not None:
            metadata["header"] = str(header).strip()

        return metadata

    @classmethod
    def to_llama_document(
            cls,
            chunk: Any
    ) -> Optional[Document]:
        """
        Convert chunk to LlamaIndex Document.
        """

        text = cls.extract_text(chunk)
        metadata = cls.extract_metadata(chunk)

        if not text.strip():
            return None

        return Document(
            text=text,
            metadata=metadata
        )


class TextTokenizer:
    """
    Simple Russian-friendly tokenizer for BM25.
    """

    TOKEN_PATTERN = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9_.]+")

    @classmethod
    def tokenize(
            cls,
            text: str
    ) -> list[str]:
        """
        Tokenize text.
        """

        return [
            token.lower()
            for token in cls.TOKEN_PATTERN.findall(text or "")
        ]


class QdrantDenseRetriever(BaseDenseRetriever):
    """
    Dense retriever based on Qdrant.
    """

    def __init__(
            self,
            vector_store: Any
    ) -> None:
        """
        Initialize dense retriever.
        """

        self.vector_store = vector_store

    def search(
            self,
            query_vec: list[float],
            k: int
    ) -> list[SearchResult]:
        """
        Search Qdrant.
        """

        hits = self.vector_store.search(
            query_vector=query_vec,
            limit=k
        )

        results: list[SearchResult] = []

        for hit in hits:
            result = SearchResult.from_qdrant(hit)

            if result.text and result.text.strip():
                result.payload = result.payload or {}
                result.payload["retrieval_source"] = "dense"

                article_number = result.payload.get("article_number")

                if article_number is not None:
                    result.payload["article_number"] = str(
                        article_number
                    ).strip()

                results.append(result)

        return results


class BM25SparseRetriever(BaseSparseRetriever):
    """
    BM25 sparse retriever over loaded chunks.
    """

    def __init__(
            self,
            chunks: Optional[list[Any]] = None
    ) -> None:
        """
        Initialize BM25 retriever.
        """

        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks:
            self.build(chunks)

    def build(
            self,
            chunks: list[Any]
    ) -> None:
        """
        Build BM25 index from chunks.
        """

        documents: list[Document] = []

        for chunk in chunks:
            document = ChunkAdapter.to_llama_document(chunk)

            if document is not None:
                documents.append(document)

        self.documents = documents

        self.tokenized_corpus = [
            TextTokenizer.tokenize(document.text)
            for document in documents
        ]

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search BM25 index.
        """

        if self.bm25 is None:
            return []

        query_tokens = TextTokenizer.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )[:k]

        results: list[SearchResult] = []

        for index in ranked_indexes:
            score = float(scores[index])

            if score <= 0:
                continue

            document = self.documents[index]
            payload = dict(document.metadata or {})
            payload["retrieval_source"] = "bm25"

            article_number = payload.get("article_number")

            if article_number is not None:
                payload["article_number"] = str(article_number).strip()

            result = SearchResultFactory.create(
                id=f"bm25:{index}",
                text=document.text,
                score=score,
                payload=payload
            )

            results.append(result)

        return results


class LlamaIndexMetadataGraphRetriever(BaseGraphRetriever):
    """
    Lightweight GraphRAG retriever based on LlamaIndex Documents metadata.

    It does not call OpenAI and does not build PropertyGraphIndex.
    It uses graph_rag metadata, article links, topics and relation targets.
    """

    def __init__(
            self,
            chunks: Optional[list[Any]] = None
    ) -> None:
        """
        Initialize metadata graph retriever.
        """

        self.documents: list[Document] = []
        self.article_to_docs: dict[str, list[int]] = {}
        self.topic_to_articles: dict[str, set[str]] = {}
        self.article_graph: dict[str, set[str]] = {}

        if chunks:
            self.build(chunks)

    def build(
            self,
            chunks: list[Any]
    ) -> None:
        """
        Build metadata graph from chunks.
        """

        self.documents = []
        self.article_to_docs = {}
        self.topic_to_articles = {}
        self.article_graph = {}

        for chunk in chunks:
            document = ChunkAdapter.to_llama_document(chunk)

            if document is None:
                continue

            index = len(self.documents)
            self.documents.append(document)

            metadata = document.metadata or {}

            article = metadata.get("article_number")

            if article is None:
                continue

            article = str(article).strip()

            self.article_to_docs.setdefault(article, []).append(index)
            self.article_graph.setdefault(article, set())

            self._index_topics(
                article=article,
                metadata=metadata
            )

            self._index_relations(
                article=article,
                metadata=metadata
            )

    def _index_topics(
            self,
            *,
            article: str,
            metadata: dict[str, Any]
    ) -> None:
        """
        Index classic_rag topics as graph concepts.
        """

        classic_rag = metadata.get("classic_rag")

        topics = []

        if isinstance(classic_rag, dict):
            topics = classic_rag.get("topics") or []

        direct_topics = metadata.get("topics") or []

        if isinstance(direct_topics, list):
            topics.extend(direct_topics)

        for topic in topics:
            topic_key = str(topic).lower().strip()

            if topic_key:
                self.topic_to_articles.setdefault(
                    topic_key,
                    set()
                ).add(article)

    def _index_relations(
            self,
            *,
            article: str,
            metadata: dict[str, Any]
    ) -> None:
        """
        Index graph_rag relations.
        """

        graph_rag = metadata.get("graph_rag")

        if not isinstance(graph_rag, dict):
            return

        relations = graph_rag.get("relations") or []

        if not isinstance(relations, list):
            return

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            target = relation.get("target")

            if target is None:
                continue

            target_text = str(target).strip()

            if not target_text:
                continue

            target_article = self._extract_article_number(target_text)

            if target_article:
                self.article_graph.setdefault(
                    article,
                    set()
                ).add(target_article)

    @staticmethod
    def _extract_article_number(
            value: str
    ) -> Optional[str]:
        """
        Extract article number from ids like tk_rf_article_133_1.
        """

        match = re.search(r"article_(\d+)(?:_(\d+))?$", value)

        if not match:
            return None

        first = match.group(1)
        second = match.group(2)

        if second is not None:
            return f"{first}.{second}"

        return first

    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Search graph by query concepts and related articles.
        """

        if not self.documents:
            return []

        query_tokens = set(
            TextTokenizer.tokenize(query)
        )

        article_scores: dict[str, float] = {}

        for topic, articles in self.topic_to_articles.items():
            topic_tokens = set(
                TextTokenizer.tokenize(topic)
            )

            if not topic_tokens:
                continue

            overlap = len(query_tokens & topic_tokens)

            if overlap <= 0:
                continue

            for article in articles:
                article_scores[article] = (
                    article_scores.get(article, 0.0)
                    + overlap
                )

        for article in list(article_scores.keys()):
            related_articles = self.article_graph.get(article, set())

            for related in related_articles:
                article_scores[related] = (
                    article_scores.get(related, 0.0)
                    + article_scores[article] * 0.5
                )

        ranked_articles = sorted(
            article_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results: list[SearchResult] = []

        for article, article_score in ranked_articles:
            doc_indexes = self.article_to_docs.get(article, [])

            for doc_index in doc_indexes:
                document = self.documents[doc_index]

                payload = dict(document.metadata or {})
                payload["article_number"] = str(article)
                payload["retrieval_source"] = "graph"

                results.append(
                    SearchResultFactory.create(
                        id=f"graph:{article}:{doc_index}",
                        text=document.text,
                        score=float(article_score),
                        payload=payload
                    )
                )

                if len(results) >= k:
                    return results

        return results


class AlphaFusionService:
    """
    Alpha fusion for dense, BM25 and graph results.

    Formula:
        final_score =
            alpha * dense_score
            + (1 - alpha) * bm25_score
            + graph_weight * graph_score
    """

    def __init__(
            self,
            config: HybridRetrieverConfig
    ) -> None:
        """
        Initialize fusion service.
        """

        self.config = config

    def fuse(
            self,
            *,
            dense_results: list[SearchResult],
            bm25_results: list[SearchResult],
            graph_results: list[SearchResult],
            top_k: int
    ) -> list[SearchResult]:
        """
        Fuse retrieval results.
        """

        dense_results = self._normalize_scores(dense_results)
        bm25_results = self._normalize_scores(bm25_results)
        graph_results = self._normalize_scores(graph_results)

        merged: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}

        self._add_results(
            merged=merged,
            scores=scores,
            results=dense_results,
            weight=self.config.alpha,
            source="dense"
        )

        self._add_results(
            merged=merged,
            scores=scores,
            results=bm25_results,
            weight=1.0 - self.config.alpha,
            source="bm25"
        )

        self._add_results(
            merged=merged,
            scores=scores,
            results=graph_results,
            weight=self.config.graph_weight,
            source="graph"
        )

        fused = list(merged.values())

        for result in fused:
            key = self._dedupe_key(result)
            result.score = scores.get(key, 0.0)

        filtered = self._basic_filter(fused)

        filtered.sort(
            key=lambda item: item.score,
            reverse=True
        )

        return filtered[:top_k]

    def _add_results(
            self,
            *,
            merged: dict[str, SearchResult],
            scores: dict[str, float],
            results: list[SearchResult],
            weight: float,
            source: str
    ) -> None:
        """
        Add weighted source results.
        """

        for result in results:
            key = self._dedupe_key(result)
            weighted_score = float(result.score or 0.0) * weight

            if key not in merged:
                merged[key] = result
                scores[key] = weighted_score

                payload = merged[key].payload or {}
                payload["retrieval_sources"] = [source]
                merged[key].payload = payload

            else:
                scores[key] += weighted_score

                merged[key].payload = self._merge_payloads(
                    merged[key].payload or {},
                    result.payload or {},
                    source
                )

    def _normalize_scores(
            self,
            results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Normalize scores to 0..1.
        """

        if not results:
            return []

        raw_scores = [
            float(result.score or 0.0)
            for result in results
        ]

        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if math.isclose(max_score, min_score):
            for result in results:
                result.score = 1.0

            return results

        for result in results:
            result.score = (
                float(result.score or 0.0)
                - min_score
            ) / (
                max_score
                - min_score
            )

        return results

    def _basic_filter(
            self,
            hits: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Remove empty, short and duplicate results.
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
    def _dedupe_key(
            result: SearchResult
    ) -> str:
        """
        Build stable dedupe key.
        """

        payload = result.payload or {}

        article = payload.get("article_number")
        header = payload.get("header")

        if article and header:
            return f"article:{article}|header:{header}"

        if getattr(result, "id", None):
            return str(result.id)

        text = (getattr(result, "text", "") or "")[:500]

        return hashlib.md5(
            text.encode()
        ).hexdigest()

    @staticmethod
    def _merge_payloads(
            left: dict[str, Any],
            right: dict[str, Any],
            source: str
    ) -> dict[str, Any]:
        """
        Merge payloads.
        """

        merged = dict(left)

        for key, value in right.items():
            if key not in merged:
                merged[key] = value

        sources = set(merged.get("retrieval_sources", []))
        sources.add(source)

        existing_source = merged.get("retrieval_source")

        if existing_source:
            sources.add(existing_source)

        merged["retrieval_sources"] = sorted(sources)

        return merged


class Retriever(BaseRetriever):
    """
    Hybrid Retriever:
    - Dense Qdrant
    - BM25 sparse
    - metadata GraphRAG via LlamaIndex Documents
    - AlphaFusion
    """

    def __init__(
            self,
            vector_store: Any,
            embedder: Embedder,
            *,
            chunks: Optional[list[Any]] = None,
            config: Optional[HybridRetrieverConfig] = None
    ) -> None:
        """
        Initialize hybrid retriever.
        """

        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or HybridRetrieverConfig()

        self.dense = QdrantDenseRetriever(vector_store)
        self.bm25 = BM25SparseRetriever(chunks)
        self.graph = LlamaIndexMetadataGraphRetriever(chunks)
        self.fusion = AlphaFusionService(self.config)

    def build_sparse_and_graph(
            self,
            chunks: list[Any]
    ) -> None:
        """
        Build BM25 and graph indexes after ingestion.
        """

        self.bm25.build(chunks)
        self.graph.build(chunks)

    def retrieve(
            self,
            query: str,
            top_k: int = 10
    ) -> list[SearchResult]:
        """
        Retrieve using Dense + BM25 + GraphRAG.
        """

        query = (query or "").strip()

        if not query:
            return []

        pool_size = min(
            self.config.max_pool_size,
            max(
                top_k * self.config.pool_multiplier,
                30
            )
        )

        query_vec = self.embedder.encode_queries(
            [query]
        )[0]

        dense_candidates = self.dense.search(
            query_vec=query_vec,
            k=pool_size
        )

        bm25_candidates = self.bm25.search(
            query=query,
            k=pool_size
        )

        graph_candidates = self.graph.search(
            query=query,
            k=pool_size
        )

        return self.fusion.fuse(
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            graph_results=graph_candidates,
            top_k=top_k
        )

    def debug_query(
            self,
            query: str,
            top_k: int = 10
    ) -> None:
        """
        Print hybrid debug output.
        """

        print("\n" + "=" * 100)
        print("[HYBRID RETRIEVAL DEBUG: DENSE + BM25 + GRAPH]")
        print(f"QUERY: {query}")
        print("=" * 100)

        hits = self.retrieve(
            query=query,
            top_k=top_k
        )

        if not hits:
            print("No hits")
            return

        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}

            print("\n" + "-" * 100)
            print(f"TOP {index}")
            print(f"SCORE   : {hit.score:.4f}")
            print(f"SOURCES : {payload.get('retrieval_sources')}")
            print(f"ARTICLE : {payload.get('article_number', 'unknown')}")
            print(f"HEADER  : {payload.get('header', 'unknown')}")
            print(f"ID      : {hit.id}")
            print("\nTEXT:\n")
            print((hit.text or "")[:1200])