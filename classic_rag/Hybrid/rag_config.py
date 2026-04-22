import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any


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
    payload: dict[str, Any]
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_id is None:
            self.chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    self.payload.get("idempotency_key", self.text[:512]),
                )
            )

        if self.payload is None:
            self.payload = {}


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
    payload: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()

        if self.payload is None:
            self.payload = {}

        self.payload.setdefault("text", self.text)

    @classmethod
    def from_qdrant(cls, point) -> "SearchResult":
        """
        Создание SearchResult из объекта Qdrant (ScoredPoint).

        Args:
            point (Any): Объект результата из Qdrant

        Returns:
            SearchResult: Унифицированный результат
        """

        payload = getattr(point, "payload", {}) or {}

        return cls(
            text=str(payload.get("text", "")).strip(),
            score=float(getattr(point, "score", 0.0)),
            payload=payload,
            id=str(getattr(point, "id", None)),
            source="qdrant",
        )

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

        return cls(
            text=text,
            score=float(score),
            payload=payload or {},
            source="bm25",
        )

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

        return cls(
            text=base.text,
            score=float(score),
            payload=base.payload,
            id=base.id,
            source="reranker",
        )