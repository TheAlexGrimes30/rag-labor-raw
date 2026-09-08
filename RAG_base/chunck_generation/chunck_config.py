import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGResponse:
    """
    Объект ответа, возвращаемый RAG-пайплайном.

    Атрибуты:
        answer (str):
            Ответ, сгенерированный LLM.

        sources (list[dict]):
            Найденные чанки-источники вместе с их метаданными.
    """

    answer: str
    sources: list[dict]


@dataclass
class ChunkMetadata:
    """
    Контейнер метаданных для чанка RAG.

    Атрибуты:

    source (str):
        Логический источник документа
        (например, "Гражданский кодекс РФ").

    file (str):
        Путь к исходному файлу или идентификатор документа.

    header (str | None):
        Заголовок раздела или подраздела,
        извлечённый из Markdown-документа.

    level (int | None):
        Уровень Markdown-заголовка (1–6),
        представляющий глубину иерархии документа.

    article_number (str | None):
        Номер юридической статьи
        (например, "307").

    chunk_index (int | None):
        Последовательный индекс чанка внутри документа.

    topics (list[str]):
        Семантические теги, используемые для
        гибридного поиска и фильтрации.
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
    Представляет один текстовый чанк в RAG-пайплайне.

    Chunk является основной единицей индексации
    и поиска в системе.

    Он объединяет исходный текст,
    структурированные метаданные
    и детерминированный уникальный идентификатор.

    Атрибуты:

        text (str):
            Исходное текстовое содержимое чанка.

        metadata (ChunkMetadata):
            Структурированные метаданные,
            используемые для фильтрации,
            ранжирования и отслеживания источника.

        chunk_id (str | None):
            Стабильный уникальный идентификатор чанка.

            Если идентификатор не был передан,
            он детерминированно генерируется
            внутри метода __post_init__.
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

            self.chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    key
                )
            )

    def to_payload(self) -> dict[str, Any]:
        """
        Преобразует чанк в плоский словарь
        для сохранения в векторной базе данных.

        Возвращает:
            dict[str, Any]:
                Сериализованное представление чанка
                вместе с его метаданными.
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

