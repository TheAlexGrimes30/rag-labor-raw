from __future__ import annotations

import json
import math

import pandas as pd


def normalize_article_id(value) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):
        text = text[:-2]

    return text


def unique_preserve_order(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def precision_at_k(retrieved, relevant, k) -> float:
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(
        1 for x in top_k
        if x in relevant_set
    )

    return hits / k


def recall_at_k(retrieved, relevant, k) -> float:
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(
        1 for x in top_k
        if x in relevant_set
    )

    return hits / len(relevant_set)


def hit_rate_at_k(retrieved, relevant, k) -> float:
    top_k = retrieved[:k]
    relevant_set = set(relevant)

    return float(
        any(x in relevant_set for x in top_k)
    )


def reciprocal_rank(retrieved, relevant) -> float:
    relevant_set = set(relevant)

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved, relevant, k) -> float:
    relevant_set = set(relevant)
    score = 0.0

    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1 if doc_id in relevant_set else 0
        score += rel / math.log2(i + 1)

    return score


def ndcg_at_k(retrieved, relevant, k) -> float:
    if not relevant:
        return 0.0

    dcg = dcg_at_k(
        retrieved=retrieved,
        relevant=relevant,
        k=k
    )

    ideal_ranking = relevant[:k]

    idcg = dcg_at_k(
        retrieved=ideal_ranking,
        relevant=relevant,
        k=k
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


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
        "recall": [],
        "hit_rate": [],
        "mrr": [],
        "ndcg": []
    }

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

        for rank, h in enumerate(hits, start=1):
            article = None

            if h.payload:
                article = h.payload.get("article_number")

            article = normalize_article_id(article)

            if article is not None:
                retrieved_articles_raw.append(article)

            retrieved_chunks.append({
                "rank": rank,
                "article": article,
                "score": round(h.score, 4),
                "text": (h.text or "")[:1500]
            })

        retrieved_articles = unique_preserve_order(
            retrieved_articles_raw
        )

        retrieved_articles = retrieved_articles[:rerank_top_n]

        precision = precision_at_k(
            retrieved=retrieved_articles,
            relevant=relevant_articles,
            k=rerank_top_n
        )

        recall = recall_at_k(
            retrieved=retrieved_articles,
            relevant=relevant_articles,
            k=rerank_top_n
        )

        hit_rate = hit_rate_at_k(
            retrieved=retrieved_articles,
            relevant=relevant_articles,
            k=rerank_top_n
        )

        mrr = reciprocal_rank(
            retrieved=retrieved_articles,
            relevant=relevant_articles
        )

        ndcg = ndcg_at_k(
            retrieved=retrieved_articles,
            relevant=relevant_articles,
            k=rerank_top_n
        )

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["hit_rate"].append(hit_rate)
        metrics["mrr"].append(mrr)
        metrics["ndcg"].append(ndcg)

        print(f"RELEVANT: {relevant_articles}")
        print(f"HARD NEGATIVES: {hard_negatives}")
        print(f"RETRIEVED UNIQUE: {retrieved_articles}")
        print(f"PRECISION@{rerank_top_n}: {precision:.4f}")
        print(f"RECALL@{rerank_top_n}: {recall:.4f}")
        print(f"HITRATE@{rerank_top_n}: {hit_rate:.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"nDCG@{rerank_top_n}: {ndcg:.4f}")

        results.append({
            "question": query,
            "relevant_articles": relevant_articles,
            "hard_negatives": hard_negatives,
            "retrieved_articles": retrieved_articles,
            f"precision@{rerank_top_n}": precision,
            f"recall@{rerank_top_n}": recall,
            f"hitrate@{rerank_top_n}": hit_rate,
            "mrr": mrr,
            f"ndcg@{rerank_top_n}": ndcg,
            "retrieved_chunks": retrieved_chunks
        })

    dataset_size = len(dataset)

    report = {
        f"precision@{rerank_top_n}": round(
            sum(metrics["precision"]) / dataset_size,
            4
        ),
        f"recall@{rerank_top_n}": round(
            sum(metrics["recall"]) / dataset_size,
            4
        ),
        f"hitrate@{rerank_top_n}": round(
            sum(metrics["hit_rate"]) / dataset_size,
            4
        ),
        "mrr": round(
            sum(metrics["mrr"]) / dataset_size,
            4
        ),
        f"ndcg@{rerank_top_n}": round(
            sum(metrics["ndcg"]) / dataset_size,
            4
        )
    }

    print("\n" + "=" * 100)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 100)

    print(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        )
    )

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
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4
        )

    pd.DataFrame(results).to_csv(
        output_path.replace(".json", ".csv"),
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved -> {output_path}")

    return output