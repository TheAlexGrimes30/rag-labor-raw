from __future__ import annotations

import json
import math
from typing import Any, List

import pandas as pd

def unique_preserve_order(items: List[int]) -> List[int]:
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_set = set(relevant)
    hits = len([x for x in top_k if x in relevant_set])

    return hits / len(top_k)


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = len([x for x in top_k if x in relevant_set])

    return hits / len(relevant_set)


def evaluate_rag(
    rag,
    dataset,
    output_path="rag_eval_results.json",
    use_reranker=True,
    retrieve_top_k=20,
    rerank_top_n=5
):

    results = []

    metrics = {
        "precision": [],
        "recall": []
    }

    print("\n" + "=" * 100)
    print("STARTING DATASET EVALUATION")
    print("=" * 100)

    print(f"\nRERANKER ENABLED: {use_reranker}")
    print(f"RETRIEVE TOP-K: {retrieve_top_k}")
    print(f"FINAL TOP-N: {rerank_top_n}")

    for item in dataset:

        query = item["question"]
        relevant_articles = item.get("relevant_articles", [])
        hard_negatives = item.get("hard_negatives", [])

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

        for rank, h in enumerate(hits, start=1):

            article = None
            if h.payload:
                article = h.payload.get("article_number")

            try:
                if article is not None:
                    article = int(article)
                    retrieved_articles_raw.append(article)
            except Exception:
                continue

            retrieved_chunks.append({
                "rank": rank,
                "article": article,
                "score": round(h.score, 4),
                "text": (h.text or "")[:1500]
            })

        retrieved_articles = unique_preserve_order(retrieved_articles_raw)
        retrieved_articles = retrieved_articles[:rerank_top_n]

        prec = precision_at_k(retrieved_articles, relevant_articles, rerank_top_n)
        rec = recall_at_k(retrieved_articles, relevant_articles, rerank_top_n)

        metrics["precision"].append(prec)
        metrics["recall"].append(rec)

        print(f"RELEVANT: {relevant_articles}")
        print(f"RETRIEVED UNIQUE: {retrieved_articles}")

        print(f"PRECISION@{rerank_top_n}: {prec:.4f}")
        print(f"RECALL@{rerank_top_n}: {rec:.4f}")

        results.append({
            "question": query,
            "relevant_articles": relevant_articles,
            "hard_negatives": hard_negatives,
            "retrieved_articles": retrieved_articles,
            f"precision@{rerank_top_n}": prec,
            f"recall@{rerank_top_n}": rec,
            "retrieved_chunks": retrieved_chunks
        })

    report = {
        f"precision@{rerank_top_n}": round(sum(metrics["precision"]) / len(dataset), 4),
        f"recall@{rerank_top_n}": round(sum(metrics["recall"]) / len(dataset), 4),
    }

    print("\n" + "=" * 100)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 100)

    print(json.dumps(report, indent=4, ensure_ascii=False))

    output = {
        "config": {
            "use_reranker": use_reranker,
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_n": rerank_top_n
        },
        "retrieval_metrics": report,
        "samples": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    pd.DataFrame(results).to_csv(
        output_path.replace(".json", ".csv"),
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved -> {output_path}")