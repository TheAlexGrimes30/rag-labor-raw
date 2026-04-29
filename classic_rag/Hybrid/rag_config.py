import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal

from qdrant_client.http.models import PointStruct


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]


@dataclass
class Chunk:
    """
    Единица данных (чанк) в RAG-пайплайне.

    Используется на всех этапах:
    - разбиение документов (chunking)
    - генерация эмбеддингов
    - загрузка в векторное хранилище

    Attributes:
        text (str): Текст чанка
        payload (Dict[str, Any]): Метаданные
        chunk_id (str): Уникальный ID (генерируется автоматически, если не задан)
    """

    text: str
    payload: Dict[str, Any]
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}

        if self.chunk_id is None:
            key = self.payload.get("idempotency_key") or self.text[:512]

            self.chunk_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, key)
            )


@dataclass
class SearchResult:
    """
    Унифицированный результат поиска.

    Абстрагирует различные источники:
    - Qdrant (dense retrieval)
    - BM25 (sparse retrieval)
    - Reranker (cross-encoder)

    Это позволяет:
    - не зависеть от конкретного backend
    - легко менять реализацию retrieval

    Attributes:
        text (str): Текст
        score (float): Оценка релевантности
        payload (Dict[str, Any]): Метаданные
        id (Optional[str]): Идентификатор
        source (Optional[str]): Источник результата
    """

    text: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}

        self.text = (self.text or "").strip()
        self.payload.setdefault("text", self.text)

    @classmethod
    def from_qdrant(cls, point: PointStruct) -> "SearchResult":
        """
        Создание SearchResult из объекта Qdrant (ScoredPoint).

        Args:
            point (Any): Объект результата из Qdrant

        Returns:
            SearchResult: Унифицированный результат
        """

        return QdrantMapper.map(point)

    @classmethod
    def from_bm25(cls, text: str, score: float, payload: dict[str, Any] | None = None) -> "SearchResult":
        """
        Создание результата из BM25.

        Args:
            text (str): Текст документа
            score (float): BM25 score
            payload (Optional[Dict[str, Any]]): Метаданные

        Returns:
            SearchResult: Унифицированный результат
        """

        return BM25Mapper.map(text, score, payload)

    @classmethod
    def from_rerank(cls, base: "SearchResult", score: float) -> "SearchResult":
        """
        Создание результата после reranking.

        Args:
            base (SearchResult): Базовый результат (до rerank)
            score (float): Новый score после cross-encoder

        Returns:
            SearchResult: Обновлённый результат
        """

        return RerankMapper.map(base, score)

class QdrantMapper:
    """
    Преобразует результат Qdrant → SearchResult
    """

    @classmethod
    def map(cls, point: PointStruct) -> SearchResult:
        payload = getattr(point, "payload", {}) or {}
        raw_id = getattr(point, "id", None)

        return SearchResult(
            text=str(payload.get("text", "")),
            score=float(getattr(point, "score", 0.0)),
            payload=payload,
            id=str(raw_id) if raw_id is not None else None,
            source="qdrant",
        )


class BM25Mapper:
    """
    Преобразует результат BM25 → SearchResult
    """

    @classmethod
    def map(
        cls,
        text: str,
        score: float,
        payload: dict[str, Any] | None = None
    ) -> SearchResult:
        return SearchResult(
            text=text or "",
            score=float(score),
            payload=payload or {},
            source="bm25",
        )

class RerankMapper:
    """
    Преобразует результат после reranking
    """

    @classmethod
    def map(cls, base: SearchResult, score: float) -> SearchResult:
        return SearchResult(
            text=base.text,
            score=float(score),
            payload=base.payload.copy(),
            id=base.id,
            source=f"{base.source}+reranker" if base.source else "reranker",
        )

@dataclass
class ChunkMetadata:
    source: str
    file: str

    chunk_type: Literal["structure", "sentence", "window"]

    header: str | None
    level: int | None
    article_number: str | None

    topics: list[str]
    