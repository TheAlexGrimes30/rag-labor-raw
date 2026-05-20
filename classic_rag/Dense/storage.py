from dataclasses import dataclass
from typing import List, Optional, Any, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)


@dataclass
class VectorStore:
    """
    Thin wrapper over Qdrant vector database.

    Provides:
    - collection management
    - batch upsert of embeddings
    - vector similarity search
    - payload normalization
    """

    client: QdrantClient
    collection_name: str
    vector_size: int
    distance: Distance = Distance.COSINE


    def ensure_collection(self) -> None:
        """
        Ensure that Qdrant collection exists.

        If collection does not exist, it is created
        with predefined vector size and distance metric.
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
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        batch_size: int = 64,
    ) -> None:
        """
        Insert or update embeddings in Qdrant.

        Args:
            ids (List[str]):
                Unique chunk/document identifiers.

            vectors (List[List[float]]):
                Embedding vectors corresponding to documents.

            payloads (List[dict]):
                Metadata payloads for each vector.

            batch_size (int):
                Batch size for Qdrant upsert requests.

        Raises:
            ValueError:
                If input data is empty or invalid.
        """

        if not ids or not vectors:
            raise ValueError("[VectorStore] Empty ids or vectors")

        points: List[PointStruct] = []

        for i, v, p in zip(ids, vectors, payloads):

            if not v or len(v) != self.vector_size:
                continue

            points.append(
                PointStruct(
                    id=str(i),
                    vector=v,
                    payload=self._normalize_payload(p),
                )
            )

        if not points:
            raise ValueError("[VectorStore] No valid points to upsert")

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )


    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
    ) -> list[Any]:
        """
        Perform similarity search in Qdrant collection.

        Args:
            query_vector (List[float]):
                Query embedding vector.

            limit (int):
                Number of nearest neighbors to return.

            uery_filter (Optional[Filter]):
                    Optional Qdrant filter for metadata filtering.

        Returns:
            List[Any]:
                Search results (Qdrant points).
        """

        if not query_vector:
            return []

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return result.points if hasattr(result, "points") else result


    def delete_collection(self) -> None:
        """
        Delete Qdrant collection if it exists.
        """

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)


    def _normalize_payload(self, p: Optional[dict]) -> dict:
        """
        Normalize metadata payload before storing in Qdrant.

        Ensures consistent schema across all vectors.

        Args:
            p (Optional[Dict]):
                Raw payload dictionary.

        Returns:
            Dict:
                Normalized payload.
        """

        p = p or {}

        return {
            "text": p.get("text", "") or "",
            "source": p.get("source"),
            "file": p.get("file"),
            "header": p.get("header"),
            "level": p.get("level"),
            "article_number": p.get("article_number"),
            "topics": p.get("topics") or [],
        }