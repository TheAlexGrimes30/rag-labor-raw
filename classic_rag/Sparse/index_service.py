from typing import List

from classic_rag.Sparse.rag_config import Chunk
from classic_rag.Sparse.retriever import BM25Retriever


class IndexService:

    def __init__(self):
        self.documents = []
        self.retriever = None


    def index(self, chunks: List[Chunk]) -> None:

        if not chunks:
            return

        self.documents = chunks

        self.retriever = BM25Retriever(
            documents=chunks
        )

        print(f"[Index] BM25 indexed: {len(chunks)} chunks")

    def rebuild(self, chunks: List[Chunk]) -> None:
        self.index(chunks)

    def get_retriever(self) -> BM25Retriever:
        return self.retriever