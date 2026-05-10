# graph_search_result.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from qdrant_client.http.models import PointStruct


@dataclass
class SearchResult:

    text: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    node_id: Optional[str] = None
    edges: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}

        self.text = (self.text or "").strip()
        self.payload.setdefault("text", self.text)
        if self.node_id:
            self.payload.setdefault("node_id", self.node_id)
        if self.edges:
            self.payload.setdefault("edges", self.edges)

    @classmethod
    def from_qdrant(cls, point: PointStruct) -> "SearchResult":
        return GraphQdrantMapper.map(point)

    @classmethod
    def from_bm25(cls, text: str, score: float, payload: Optional[Dict[str, Any]] = None) -> "SearchResult":
        return GraphBM25Mapper.map(text, score, payload)

    @classmethod
    def from_rerank(cls, base: "SearchResult", score: float) -> "SearchResult":
        return GraphRerankMapper.map(base, score)


class GraphQdrantMapper:
    @classmethod
    def map(cls, point: PointStruct) -> SearchResult:
        payload = getattr(point, "payload", {}) or {}
        return SearchResult(
            text=str(payload.get("text", "")),
            score=float(point.score),
            payload=payload,
            id=str(point.id) if point.id else None,
            source="qdrant",
            node_id=payload.get("node_id"),
            edges=payload.get("edges", [])
        )


class GraphBM25Mapper:
    @classmethod
    def map(cls, text: str, score: float, payload: Optional[Dict[str, Any]] = None) -> SearchResult:
        payload = payload or {}
        return SearchResult(
            text=text or "",
            score=float(score),
            payload=payload,
            source="bm25",
            node_id=payload.get("node_id"),
            edges=payload.get("edges", [])
        )


class GraphRerankMapper:
    @classmethod
    def map(cls, base: SearchResult, score: float) -> SearchResult:
        return SearchResult(
            text=base.text,
            score=float(score),
            payload=base.payload.copy(),
            id=base.id,
            source=f"{base.source}+reranker" if base.source else "reranker",
            node_id=base.node_id,
            edges=base.edges
        )
    