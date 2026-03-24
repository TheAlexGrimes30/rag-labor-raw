from abc import ABC, abstractmethod
from typing import List

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        pass


class DenseRetriever(BaseRetriever):

    def __init__(self, documents: List[Document]):
        self.documents = documents

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        self.db = FAISS.from_documents(documents, self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        query = "query: " + query
        return self.db.similarity_search(query, k=k)

class SparseRetriever(BaseRetriever):

    def __init__(self, documents: List[Document]):
        self.documents = documents

        self.corpus = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:k]

        return [self.documents[i] for i in top_k_idx]

class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        documents: List[Document],
        alpha: float = 0.6
    ):
        self.documents = documents
        self.alpha = alpha

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        self.db = FAISS.from_documents(documents, self.embeddings)

        self.corpus = [
            doc.page_content.lower().split()
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:

        dense_results = self.db.similarity_search_with_score(
            "query: " + query, k=len(self.documents)
        )

        dense_scores = {
            doc.page_content: score
            for doc, score in dense_results
        }

        tokenized_query = query.lower().split()
        sparse_scores_array = self.bm25.get_scores(tokenized_query)

        sparse_scores = {
            self.documents[i].page_content: sparse_scores_array[i]
            for i in range(len(self.documents))
        }

        def normalize(scores_dict):
            values = list(scores_dict.values())
            min_v, max_v = min(values), max(values)

            return {
                k: (v - min_v) / (max_v - min_v + 1e-8)
                for k, v in scores_dict.items()
            }

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        combined_scores = {}

        for doc in self.documents:
            content = doc.page_content

            d = dense_norm.get(content, 0)
            s = sparse_norm.get(content, 0)

            combined_scores[content] = (
                self.alpha * d + (1 - self.alpha) * s
            )

        ranked_docs = sorted(
            self.documents,
            key=lambda d: combined_scores[d.page_content],
            reverse=True
        )

        return ranked_docs[:k]

