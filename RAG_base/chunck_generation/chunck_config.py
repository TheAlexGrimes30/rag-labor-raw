import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGResponse:
    """
    Объект ответа, возвращаемый RAG-пайплайном
    """

    answer: str

@dataclass
class ChunkMetadata:
    """
    Метаданные чанка юридического RAG.
    """

    source: str
    file: str

    header: str | None = None
    level: int | None = None
    article_number: str | None = None

    chunk_index: int | None = None
    topics: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """
    Chunk юридической системы
    """

    text: str
    metadata: ChunkMetadata
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        """
        Выполняет дополнительную инициализацию
        после автоматического вызова __init__.

        Если текст чанка отсутствует,
        идентификатор не создаётся.

        Если chunk_id не был передан,
        создаётся детерминированный UUID
        на основе метаданных и текста чанка.
        """

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
        Преобразует чанк в плоский словарь
        для сохранения в векторной базе данных.
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

