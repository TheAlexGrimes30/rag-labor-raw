import uuid
from dataclasses import field, dataclass
from typing import List, Dict, Any


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

    @property
    def id(self) -> str | None:
        return self.chunk_id

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
