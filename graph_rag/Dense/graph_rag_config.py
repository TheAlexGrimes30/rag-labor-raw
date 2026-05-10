import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ChunkMetadata:
    source: str
    file: str
    header: Optional[str] = None
    level: Optional[int] = None
    article_number: Optional[str] = None
    chunk_index: Optional[int] = None
    topics: List[str] = field(default_factory=list)
    node_id: Optional[str] = None

@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata
    chunk_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.text:
            return

        if not self.chunk_id:
            key = "|".join([
                self.metadata.source or "",
                self.metadata.file or "",
                str(self.metadata.article_number or ""),
                str(self.metadata.header or ""),
                str(self.metadata.node_id or ""),
                self.text[:400]
            ])
            self.chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "text": self.text,
            "source": self.metadata.source,
            "file": self.metadata.file,
            "header": self.metadata.header,
            "level": self.metadata.level,
            "article_number": self.metadata.article_number,
            "topics": self.metadata.topics,
            "node_id": self.metadata.node_id
        }
        return payload


@dataclass
class GraphEdge:
    relation: str
    target: str

@dataclass
class GraphNode:
    node_id: str
    article_number: Optional[str] = None
    title: str = ""
    text: str = ""
    edges: List[GraphEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_edge(self, relation: str, target_id: str):
        self.edges.append(GraphEdge(relation=relation, target=target_id))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.metadata.get("source"),
            "file": self.metadata.get("file"),
            "header": self.metadata.get("header"),
            "level": self.metadata.get("level"),
            "article_number": self.article_number,
            "topics": self.metadata.get("topics", []),
            "node_id": self.node_id,
            "edges": [{"relation": e.relation, "target": e.target} for e in self.edges]
        }


def make_node_id(article_number: Optional[str], header: Optional[str], text: str) -> str:
    key = f"{article_number or ''}|{header or ''}|{text[:400]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
