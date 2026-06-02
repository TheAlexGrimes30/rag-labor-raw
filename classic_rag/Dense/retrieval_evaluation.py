from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


class ArticleIdNormalizer:
    """
    Normalizes article ids for metric calculation.

    The important case is article 133.1: if retrieval payload says 133,
    but the text is clearly about regional minimum wage, evaluation restores
    it to 133.1 so metrics are not falsely pessimistic.
    """

    REGIONAL_MIN_WAGE_MARKERS = (
        "региональная минимальная заработная плата",
        "региональной минимальной заработной плате",
        "регионального соглашения о минимальной заработной плате",
        "субъекте российской федерации может устанавливаться",
        "трехсторонней комиссией",
        "работодателям присоединиться",
        "мотивированный отказ",
    )

    @classmethod
    def normalize(
            cls,
            value: Any,
            *,
            text: str = ""
    ) -> str | None:
        """
        Convert article id to a stable string.
        """

        if value is None:
            return None

        article = str(value).strip()

        if not article:
            return None

        if article.endswith(".0"):
            article = article[:-2]

        if article in {"133_1", "133-1"}:
            return "133.1"

        if article == "133.1":
            return article

        if article == "133" and cls._looks_like_article_133_1(text):
            return "133.1"

        return article

    @classmethod
    def _looks_like_article_133_1(cls, text: str) -> bool:
        """
        Detect chunks that belong to article 133.1 by text content.
        """

        haystack = (text or "").lower()

        if "133.1" in haystack or "133_1" in haystack:
            return True

        return any(marker in haystack for marker in cls.REGIONAL_MIN_WAGE_MARKERS)


def normalize_article_id(value: Any, text: str = "") -> str | None:
    """
    Backward-compatible wrapper for article id normalization.
    """

    return ArticleIdNormalizer.normalize(value, text=text)


def unique_preserve_order(items: list[str]) -> list[str]:
    """
    Remove duplicates while preserving the first occurrence order.
    """

    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K = relevant hits in top-K / K.
    """

    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(1 for x in top_k if x in relevant_set)

    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@K = found relevant items / all relevant items.
    """

    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(1 for x in top_k if x in relevant_set)

    return hits / len(relevant_set)


def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    HitRate@K = whether at least one relevant item is present in top-K.
    """

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    return float(any(x in relevant_set for x in top_k))


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """
    Reciprocal rank of the first relevant item.
    """

    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Discounted cumulative gain with binary relevance.
    """

    relevant_set = set(relevant)
    score = 0.0

    for index, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1 if doc_id in relevant_set else 0
        score += rel / math.log2(index + 1)

    return score


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Normalized DCG@K with binary relevance.
    """

    if not relevant:
        return 0.0

    dcg = dcg_at_k(retrieved=retrieved, relevant=relevant, k=k)

    ideal_ranking = relevant[:k]
    idcg = dcg_at_k(retrieved=ideal_ranking, relevant=relevant, k=k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_metrics(
        retrieved: list[str],
        relevant: list[str],
        k: int
) -> dict[str, float]:
    """
    Calculate standard retrieval metrics for a retrieved article-id sequence.
    """

    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"hitrate@{k}": hit_rate_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


def mean_metric(samples: list[dict[str, float]], key: str) -> float:
    """
    Calculate rounded mean metric value.
    """

    if not samples:
        return 0.0

    return round(sum(sample[key] for sample in samples) / len(samples), 4)


def build_report(
        metric_samples: list[dict[str, float]],
        k: int
) -> dict[str, float]:
    """
    Build aggregate metric report.
    """

    keys = [
        f"precision@{k}",
        f"recall@{k}",
        f"hitrate@{k}",
        "mrr",
        f"ndcg@{k}",
    ]

    return {
        key: mean_metric(metric_samples, key)
        for key in keys
    }


def evaluate_rag(
        rag,
        dataset,
        output_path="rag_eval_results.json",
        use_reranker=True,
        retrieve_top_k=20,
        rerank_top_n=5
):
    """
    Evaluate RAG retrieval.

    Produces two metric groups:
    - article_level_metrics: duplicates of the same article are collapsed.
    - chunk_level_metrics: exact top-N returned chunks are evaluated as-is.

    The old `retrieval_metrics` field is kept as an alias to article-level
    metrics for backward compatibility.
    """

    results = []
    article_metric_samples = []
    chunk_metric_samples = []

    print("\n" + "=" * 100)
    print("STARTING DATASET EVALUATION")
    print("=" * 100)

    print(f"\nRERANKER ENABLED: {use_reranker}")
    print(f"RETRIEVE TOP-K: {retrieve_top_k}")
    print(f"FINAL TOP-N: {rerank_top_n}")

    for item in dataset:
        query = item["question"]

        relevant_articles = [
            normalize_article_id(x)
            for x in item.get("relevant_articles", [])
        ]

        relevant_articles = [
            x for x in relevant_articles
            if x is not None
        ]

        hard_negatives = [
            normalize_article_id(x)
            for x in item.get("hard_negatives", [])
        ]

        hard_negatives = [
            x for x in hard_negatives
            if x is not None
        ]

        print("\n" + "-" * 100)
        print(f"QUERY: {query}")

        hits = rag.search(
            query=query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_n=rerank_top_n,
            use_reranker=use_reranker
        )

        retrieved_articles_raw = []
        retrieved_chunks = []

        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            text = hit.text or ""

            raw_article = payload.get("article_number")
            article = normalize_article_id(raw_article, text=text)

            if article is not None:
                retrieved_articles_raw.append(article)

            retrieved_chunks.append({
                "rank": rank,
                "article": article,
                "raw_article": raw_article,
                "score": round(hit.score, 4),
                "sources": payload.get("retrieval_sources")
                           or payload.get("retrieval_source"),
                "header": payload.get("header"),
                "text": text[:1500]
            })

        chunk_level_articles = retrieved_articles_raw[:rerank_top_n]

        article_level_articles = unique_preserve_order(
            retrieved_articles_raw
        )[:rerank_top_n]

        article_metrics = calculate_metrics(
            retrieved=article_level_articles,
            relevant=relevant_articles,
            k=rerank_top_n
        )

        chunk_metrics = calculate_metrics(
            retrieved=chunk_level_articles,
            relevant=relevant_articles,
            k=rerank_top_n
        )

        article_metric_samples.append(article_metrics)
        chunk_metric_samples.append(chunk_metrics)

        print(f"RELEVANT: {relevant_articles}")
        print(f"HARD NEGATIVES: {hard_negatives}")
        print(f"RETRIEVED CHUNK-LEVEL: {chunk_level_articles}")
        print(f"RETRIEVED ARTICLE-LEVEL: {article_level_articles}")

        print("ARTICLE-LEVEL METRICS:")
        print(json.dumps(article_metrics, ensure_ascii=False, indent=4))

        print("CHUNK-LEVEL METRICS:")
        print(json.dumps(chunk_metrics, ensure_ascii=False, indent=4))

        results.append({
            "question": query,
            "relevant_articles": relevant_articles,
            "hard_negatives": hard_negatives,
            "retrieved_articles": article_level_articles,
            "retrieved_articles_chunk_level": chunk_level_articles,
            "article_level_metrics": article_metrics,
            "chunk_level_metrics": chunk_metrics,
            "retrieved_chunks": retrieved_chunks
        })

    article_report = build_report(
        article_metric_samples,
        rerank_top_n
    )

    chunk_report = build_report(
        chunk_metric_samples,
        rerank_top_n
    )

    print("\n" + "=" * 100)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 100)

    print("ARTICLE-LEVEL:")
    print(json.dumps(article_report, indent=4, ensure_ascii=False))

    print("CHUNK-LEVEL:")
    print(json.dumps(chunk_report, indent=4, ensure_ascii=False))

    output = {
        "config": {
            "use_reranker": use_reranker,
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_n": rerank_top_n
        },
        "retrieval_metrics": article_report,
        "article_level_metrics": article_report,
        "chunk_level_metrics": chunk_report,
        "samples": results
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)

    pd.DataFrame(results).to_csv(
        output_path.replace(".json", ".csv"),
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved -> {output_path}")

    return output
