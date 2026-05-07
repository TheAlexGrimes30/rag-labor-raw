import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]

@dataclass
class ChunkMetadata:
    source: str
    file: str
    header: str | None
    level: int | None
    article_number: str | None
    chunk_index: int | None = None
    topics: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text:
            return

        if not self.chunk_id:
            key = "|".join([
                self.metadata.source or "",
                self.metadata.file or "",
                str(self.metadata.article_number or ""),
                str(self.metadata.header or ""),
                self.text[:400]
            ])

            self.chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))


    def to_payload(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.metadata.source,
            "file": self.metadata.file,
            "header": self.metadata.header,
            "level": self.metadata.level,
            "article_number": self.metadata.article_number,
            "topics": self.metadata.topics,
        }

@dataclass
class SearchResult:
    text: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)

    id: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        self.text = (self.text or "").strip()



class QdrantMapper:

    @staticmethod
    def map(point) -> SearchResult:

        payload = getattr(point, "payload", {}) or {}

        score = getattr(point, "score", 0.0)

        return SearchResult(
            text=payload.get("text"),
            score=float(score),
            payload=payload,
            id=str(getattr(point, "id", None)),
            source="qdrant"
        )


class RerankMapper:

    @staticmethod
    def map(base: SearchResult, score: float) -> SearchResult:

        return SearchResult(
            text=base.text,
            score=float(score),
            payload=base.payload,
            id=base.id,
            source="reranker"
        )
