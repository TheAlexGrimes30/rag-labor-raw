import numpy as np
from typing import List, Dict

from classic_rag.Hybrid.rag_config import SearchResult


class Embedder:

    def __init__(self, model_name: str, batch_size: int = 16, normalize: bool = True):
        print(f"[Embedder] Loading model: {model_name}")

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()

        print(f"[Embedder] Dimension: {self.dim}")

    def encode_queries(self, texts: List[str]):
        return self._encode([f"query: {t}" for t in texts])

    def encode_passages(self, texts: List[str]):
        return self._encode([f"passage: {t}" for t in texts])

    def _encode(self, texts: List[str]):
        all_vecs = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            vecs = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            all_vecs.append(vecs)

        return np.vstack(all_vecs)

class DenseRetriever:

    def __init__(self, vector_store):
        self.vs = vector_store

    def search(self, vec, k: int) -> List[SearchResult]:

        hits = self.vs.search(vec, limit=k)

        results = []

        for h in hits:
            payload = h.payload or {}

            text = payload.get("text")
            if not text:
                continue

            results.append(
                SearchResult(
                    text=text,
                    score=float(getattr(h, "score", 0.0)),
                    payload=payload,
                    id=str(h.id),
                    source="qdrant"
                )
            )

        return results


# =========================
# SIMPLE RETRIEVER (FIXED)
# =========================

class Retriever:

    def __init__(self, vector_store, embedder):

        self.vs = vector_store
        self.embedder = embedder

        self.dense = DenseRetriever(vector_store)

        self.doc_map: Dict[str, SearchResult] = {}
        self.vec_map: Dict[str, np.ndarray] = {}

        self.ids: List[str] = []
        self.docs: List[str] = []

        self._built = False

    # =========================
    # BUILD INDEX
    # =========================
    def build(self):

        if self._built:
            return

        print("[Retriever] Building index...")

        points, _ = self.vs.client.scroll(
            collection_name=self.vs.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        for p in points:

            payload = p.payload or {}
            text = payload.get("text")

            if not text:
                continue

            doc_id = str(p.id)

            self.doc_map[doc_id] = SearchResult(
                text=text,
                score=0.0,
                payload=payload,
                id=doc_id,
                source="cache"
            )

            self.ids.append(doc_id)

            # важно: эмбеддинг делаем по чистому тексту
            self.docs.append(text)


        print("[Retriever] Encoding embeddings...")

        vecs = self.embedder.encode_passages(self.docs)

        for doc_id, vec in zip(self.ids, vecs):
            self.vec_map[doc_id] = vec

        self._built = True

    # =========================
    # RETRIEVE
    # =========================
    def retrieve(self, query: str, k: int = 10) -> List[SearchResult]:

        self.build()

        qvec = self.embedder.encode_queries([query])[0]

        # 🔥 ONLY DENSE SEARCH
        hits = self.dense.search(qvec, k=50)

        # optional: rerank locally by cosine (cheap MMR-lite)
        doc_vecs = []
        filtered = []

        for h in hits:
            if h.id in self.vec_map and self.vec_map[h.id] is not None:
                doc_vecs.append(self.vec_map[h.id])
                filtered.append(h)

        if not filtered:
            return hits[:k]

        sims = np.dot(doc_vecs, qvec)

        ranked = sorted(
            zip(filtered, sims),
            key=lambda x: x[1],
            reverse=True
        )

        return [h for h, _ in ranked[:k]]