import hashlib
import re
from abc import abstractmethod, ABC
from typing import List, Set

from rank_bm25 import BM25Okapi

from classic_rag.Sparse.search_result import SearchResult


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):

    def __init__(
        self,
        documents: List[SearchResult],
        min_text_len: int = 40
    ):

        self.min_text_len = min_text_len

        self.documents = [
            d for d in documents
            if d.text and len(d.text.strip()) >= min_text_len
        ]

        self.corpus = [
            self._tokenize(d.text)
            for d in self.documents
        ]

        self.bm25 = BM25Okapi(self.corpus)


    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:

        query = (query or "").strip()

        if not query:
            return []

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for i in ranked_idx[:top_k * 5]:

            doc = self.documents[i]
            score = scores[i]

            if score <= 0:
                continue

            doc.final_score = float(score)

            results.append(doc)

            if len(results) >= top_k:
                break

        return self._deduplicate(results)


    def _tokenize(self, text: str) -> List[str]:

        text = text.lower()

        return re.findall(r"\w+", text)


    def _deduplicate(
        self,
        hits: List[SearchResult]
    ) -> List[SearchResult]:

        seen: Set[str] = set()
        result: List[SearchResult] = []

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < self.min_text_len:
                continue

            key = (
                h.id
                or hashlib.md5(text[:200].encode()).hexdigest()
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(h)

        return result

    def debug_query(
        self,
        query: str,
        top_k: int = 10
    ):

        print("\n" + "=" * 80)
        print(f"[BM25 QUERY] {query}")

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        print(f"\n[TOP {top_k}]")

        for i in ranked[:top_k]:

            doc = self.documents[i]

            print(
                f"\nscore={scores[i]:.4f} | id={doc.id}"
            )

            print((doc.text or "")[:400])