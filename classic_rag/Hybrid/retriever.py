import re
from abc import abstractmethod, ABC
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from nltk import SnowballStemmer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rank_bm25 import BM25Okapi


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        pass


class HybridRetriever:

    def __init__(self, documents: List[Document], alpha: float = 0.6, reranker=None):
        print("Initializing HybridRetriever...")

        self.documents = documents
        self.alpha = alpha
        self.reranker = reranker

        self.stemmer = SnowballStemmer("russian")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "labor_law"

        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            print("Creating Qdrant collection...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )

        self.db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        if self.client.count(self.collection_name).count == 0:
            print("Uploading documents...")
            self.db.add_documents(documents)

    def _tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [self.stemmer.stem(t) for t in tokens]

    def _normalize(self, scores: dict):
        if not scores:
            return scores

        vals = list(scores.values())
        mn, mx = min(vals), max(vals)

        if abs(mx - mn) < 1e-8:
            return {k: 0 for k in scores}

        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

    def retrieve(self, query: str, k: int = 5):

        dense_results = self.db.similarity_search_with_score(
            query,
            k=min(50, len(self.documents))
        )

        if not dense_results:
            return []

        candidate_docs = [d for d, _ in dense_results]
        dense_scores = {d.page_content: s for d, s in dense_results}

        tokenized_query = self._tokenize(query)

        corpus = [
            self._tokenize(d.page_content)
            for d in candidate_docs
        ]

        bm25 = BM25Okapi(corpus)
        bm25_scores_arr = bm25.get_scores(tokenized_query)

        bm25_scores = {
            candidate_docs[i].page_content: bm25_scores_arr[i]
            for i in range(len(candidate_docs))
        }

        dense_n = self._normalize(dense_scores)
        bm25_n = self._normalize(bm25_scores)

        combined = {}

        for d in candidate_docs:
            c = d.page_content
            combined[c] = (
                self.alpha * dense_n.get(c, 0) +
                (1 - self.alpha) * bm25_n.get(c, 0)
            )

        ranked = sorted(
            candidate_docs,
            key=lambda d: combined.get(d.page_content, 0),
            reverse=True
        )

        seen = set()
        filtered = []

        for d in ranked:
            key = d.metadata.get("id") or d.metadata.get("source")
            if key not in seen:
                seen.add(key)
                filtered.append(d)

        top = filtered[:12]

        if self.reranker:
            top = self.reranker.rerank(
                query,
                top,
                top_k=max(k, 6)
            )

        return top[:k]