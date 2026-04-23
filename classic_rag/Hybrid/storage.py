from dataclasses import dataclass
from typing import List, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter


@dataclass
class VectorStore:
    """
    Абстракция над векторным хранилищем (Qdrant).

    Отвечает за:
    - создание коллекции
    - вставку (upsert) векторов
    - поиск по векторам
    - удаление коллекции

    Позволяет изолировать бизнес-логику RAG от конкретной реализации базы.
    """

    client: QdrantClient
    collection_name: str
    vector_size: int
    distance: Distance = Distance.COSINE

    def ensure_collection(self) -> None:
        """
        Проверяет наличие коллекции и создаёт её при отсутствии.

        Поведение:
            - если коллекция уже существует → ничего не делает
            - если нет → создаёт новую с заданными параметрами
        """

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

    def upsert(
            self,
            ids: List[str],
            vectors: List[List[float]],
            payloads: List[dict],
    ) -> None:
        """
        Добавляет или обновляет точки в векторном хранилище.

        Вход:
            ids (List[str]):
                Список уникальных идентификаторов точек

            vectors (List[List[float]]):
                Список векторов (эмбеддингов)
                Размер каждого вектора должен совпадать с vector_size

            payloads (List[dict]):
                Список метаданных для каждой точки
                (например: text, source, дополнительные поля)

        Выход:
            None
        """

        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
            self,
            query_vector: List[float],
            limit: int = 10,
            query_filter: Optional[Filter] = None,
    ) -> List[Any]:
        """
        Выполняет поиск ближайших векторов в коллекции.

        Вход:
            query_vector (List[float]):
                Вектор запроса (embedding)

            limit (int):
                Максимальное количество возвращаемых результатов

            query_filter (Optional[Filter]):
                Фильтр Qdrant для ограничения поиска
                (например: по source, тегам и т.д.)

        Выход:
            List[Any]:
                Список найденных объектов (ScoredPoint из Qdrant),
                содержащих:
                - id
                - score (релевантность)
                - payload (метаданные)

        Примечание:
            Возвращаемый тип оставлен как Any,
            чтобы не привязываться жёстко к Qdrant API.
        """

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return list(result.points)

    def delete_collection(self) -> None:
        """
        Удаляет коллекцию из Qdrant (если существует).

        Используется:
            - при очистке базы
            - при переинициализации индекса
        """

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            