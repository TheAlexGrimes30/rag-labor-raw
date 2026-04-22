from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder based reranker for ranking retrieved documents by relevance.

    This class uses a sentence-transformers CrossEncoder model to score
    (query, document) pairs and return the most relevant documents.
    """

    def __init__(self):
        """
        Initialize the reranker model.
        Loads a pretrained CrossEncoder model for relevance scoring.
        """

        print("Loading Cross-Encoder Reranker...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def rerank(self, query: str, docs: List[Document], top_k: int = 6) -> List[Document]:
        """
        Rerank a list of documents based on their relevance to the query.

        Args:
            query (str): The user query.
            docs (List[Document]): List of retrieved documents to rerank.
            top_k (int, optional): Number of top documents to return. Defaults to 6.

        Returns:
            List[Document]: Top-k documents sorted by relevance (descending).
        """

        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]