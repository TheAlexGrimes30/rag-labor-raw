from typing import List

from classic_rag.Dense.dense_retriever import Embedder
from classic_rag.Dense.rag_config import Chunk
from classic_rag.Dense.storage import VectorStore


class IndexService:
    """
    Service responsible for indexing document chunks into a vector database.

    Pipeline:
    1. Extract text from chunks
    2. Generate embeddings using Embedder
    3. Build payload metadata
    4. Upsert vectors into vector store (Qdrant / similar)

    This service is a write-side component of the RAG pipeline.
    It does NOT perform retrieval or ranking.
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def index(self, chunks: list[Chunk]) -> None:
        """
        Index a list of document chunks into vector storage.

        Steps:
        - Extract text from chunks
        - Convert text to embeddings
        - Attach metadata payloads
        - Upsert into vector database

        Args:
            chunks (List[Chunk]):
                Preprocessed document chunks.

        Returns:
            None
        """

        if not chunks:
            return

        texts = [c.text for c in chunks]
        payloads = [c.to_payload() for c in chunks]
        ids = [c.chunk_id for c in chunks]

        vectors = self.embedder.encode_passages(texts)

        self.vector_store.upsert(ids, vectors, payloads)

        print(f"[Index] Indexed: {len(chunks)} chunks")