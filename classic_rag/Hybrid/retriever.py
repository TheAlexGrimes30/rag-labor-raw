from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from classic_rag.Hybrid.rag_config import SearchResult
from classic_rag.Hybrid.storage import VectorStore


class Embedder:

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        batch_size: int = 16,
        normalize: bool = True
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_queries(self, texts: List[str]):
        return self._encode([f"query: {t}" for t in texts])

    def encode_passages(self, texts: List[str]):
        return self._encode([f"passage: {t}" for t in texts])

    def _encode(self, texts: List[str]):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).tolist()


class BM25Retriever:

    def __init__(self):
        self._bm25 = None
        self._ids = []
        self._corpus = []

    def build(self, corpus: List[str], ids: List[str]):
        self._corpus = corpus
        self._ids = ids
        self._bm25 = BM25Okapi([self._tokenize(t) for t in corpus])

    def search(self, query: str, k: int) -> Dict[str, float]:

        scores = self._bm25.get_scores(self._tokenize(query))

        ranked = sorted(
            zip(self._ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return {doc_id: float(score) for doc_id, score in ranked[:k]}

    def _tokenize(self, text: str):
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()


class QdrantDenseRetriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query_vec: List[float], k: int) -> List[SearchResult]:

        hits = self.vector_store.search(query_vec, limit=k)

        return [
            SearchResult.from_qdrant(h)
            for h in hits
            if h.payload.get("text")
        ]


class AlphaFusion:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense: List[SearchResult], sparse: Dict[str, float]):

        dense_rank = {str(d.id): i for i, d in enumerate(dense)}
        sparse_rank = {k: i for i, k in enumerate(sparse.keys())}

        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())

        fused = []

        for doc_id in all_ids:

            dr = dense_rank.get(doc_id, 999)
            sr = sparse_rank.get(doc_id, 999)

            score = 1 / (self.k + dr) + 1 / (self.k + sr)

            fused.append((doc_id, score))

        return sorted(fused, key=lambda x: x[1], reverse=True)


class Retriever:

    def __init__(self, vector_store: VectorStore, embedder: Embedder):

        self.vector_store = vector_store
        self.embedder = embedder

        self.dense = QdrantDenseRetriever(vector_store)
        self.sparse = BM25Retriever()
        self.fusion = AlphaFusion()

        self._corpus = []
        self._ids = []
        self._built = False

    def build_corpus(self):

        if self._built:
            return

        points, _ = self.vector_store.client.scroll(
            collection_name=self.vector_store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        corpus = []
        ids = []

        for p in points:
            payload = p.payload or {}

            text = payload.get("text", "").strip()
            if not text:
                continue

            article = payload.get("article_number", "")
            header = payload.get("header", "")

            enriched = f"""
            Статья ТК РФ {article}
            Раздел: {header}
            
            {text}
            """.strip()

            corpus.append(enriched)
            ids.append(str(p.id))

        self._corpus = corpus
        self._ids = ids

        self.sparse.build(corpus, ids)
        self._built = True

    def retrieve(self, query: str, top_k: int = 10):

        self.build_corpus()

        query_vec = self.embedder.encode_queries([query])[0]

        dense_hits = self.dense.search(query_vec, k=80)
        sparse_scores = self.sparse.search(query, k=80)

        fused = self.fusion.fuse(dense_hits, sparse_scores)

        dense_map = {str(d.id): d for d in dense_hits}

        results = []

        for doc_id, score in fused:

            if doc_id in dense_map:
                d = dense_map[doc_id]
                d.score = score
                results.append(d)

        return results[:top_k]

    def debug_query(self, query: str, top_k: int = 10):

        self.build_corpus()

        query_vec = self.embedder.encode_queries([query])[0]

        dense_hits = self.dense.search(query_vec, k=10)
        sparse_scores = self.sparse.search(query, k=10)

        print("\n[DENSE]")
        for d in dense_hits:
            print(d.id, d.score, d.text[:120])

        print("\n[BM25]")
        for k, v in list(sparse_scores.items()):
            print(k, v)

        fused = self.fusion.fuse(dense_hits, sparse_scores)

        print("\n[FUSED]")
        for doc_id, score in fused:
            print(doc_id, score)