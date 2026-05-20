import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RAGResponse:
    """
    Response object returned by RAG pipeline.

    Attributes:
        answer (str): Generated answer from LLM.
        sources (List[Dict]): Retrieved source chunks with metadata.
    """

    answer: str
    sources: list[dict]

@dataclass
class ChunkMetadata:
    """
    Metadata container for a RAG chunk.

    This structure is used across:
    - ingestion pipeline
    - vector storage (Qdrant)
    - retrieval filtering
    - reranking and evaluation

    Attributes:
        source (str):
            Logical source of the document (e.g. "Civil Code RF").

    file (str):
        File path or identifier of the original document.

    header (str | None):
        Section or subsection title extracted from Markdown.

    level (int | None):
        Markdown heading level (1–6), representing hierarchy depth.

    article_number (str | None):
        Legal article identifier (e.g. "307").

    chunk_index (int):
        Sequential index of chunk within the document.

    topics (List[str]):
        Semantic tags used for hybrid retrieval and filtering.
    """

    source: str
    file: str
    header: str | None
    level: int | None
    article_number: str | None
    chunk_index: int | None = None
    topics: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """
    Represents a single text chunk in the RAG pipeline.

    A Chunk is the основной unit of indexing and retrieval in the system.
    It combines raw text with metadata and a deterministic identifier.

    Attributes:
        text (str):
            Raw text content of the chunk.

        metadata (ChunkMetadata):
            Structured metadata used for filtering, ranking and traceability.

        chunk_id (str | None):
            Stable unique identifier for the chunk.
            If not provided, it is deterministically generated in __post_init__.
    """

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


    def to_payload(self) -> dict[str, Any]:
        """
        Converts chunk into a flat dictionary for vector DB storage.

        Returns:
            dict[str, Any]: Serializable representation of chunk.
        """

        return {
            "text": self.text,
            "source": self.metadata.source,
            "file": self.metadata.file,
            "header": self.metadata.header,
            "level": self.metadata.level,
            "article_number": self.metadata.article_number,
            "topics": self.metadata.topics,
        }
