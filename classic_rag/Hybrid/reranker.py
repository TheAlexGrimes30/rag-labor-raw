from sentence_transformers import CrossEncoder

from classic_rag.Hybrid.rag_config import RerankMapper


class Reranker:
    """
    CrossEncoder-based reranker for RAG pipeline.
    Responsible only for scoring and ordering candidates.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):

        print(f"[Reranker] Loading model: {model_name}")

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits, top_n: int = 6):
        """
        Re-ranks retrieved documents using CrossEncoder relevance scoring.
        """

        print(f"\n[Reranker] Query: {query}")
        print(f"[Reranker] Input hits: {len(hits)}")

        if not hits:
            return []

        # -------------------------
        # 1. prepare safe input pairs
        # -------------------------
        pairs = [
            (query, (h.text or "").strip())
            for h in hits
            if h.text and h.text.strip()
        ]

        if not pairs:
            return []

        # -------------------------
        # 2. model inference
        # -------------------------
        scores = list(
            self.model.predict(
                pairs,
                batch_size=32
            )
        )

        # -------------------------
        # 3. sort by score
        # -------------------------
        ranked = sorted(
            zip(hits, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # -------------------------
        # 4. map to SearchResult
        # -------------------------
        results = [
            RerankMapper.map(hit, score)
            for hit, score in ranked[:top_n]
        ]

        print(f"[Reranker] Final top_n: {len(results)}")

        return results

    def debug(self, query: str, hits):
        """
        Debug helper: prints reranked results.
        """

        ranked = self.rerank(query, hits, top_n=len(hits))

        for i, h in enumerate(ranked):

            print("\n" + "=" * 80)
            print(f"RANK {i + 1}")

            print("ARTICLE:", h.payload.get("article_number"))
            print("HEADER:", h.payload.get("header"))
            print("SOURCE:", h.source)

            print("\nTEXT:")
            print((h.text or "")[:500])
