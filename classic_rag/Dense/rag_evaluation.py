import json
from typing import List

import pandas as pd


def recall_at_k(retrieved: List[int], relevant: List[int]) -> float:
    if not relevant:
        return 0.0
    return float(len(set(retrieved) & set(relevant)) > 0)


def precision_at_k(retrieved: List[int], relevant: List[int]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for a in retrieved if a in relevant)
    return hits / len(retrieved)


def mrr_at_k(retrieved: List[int], relevant: List[int]) -> float:
    for i, a in enumerate(retrieved):
        if a in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_rag(rag, dataset, output_path="rag_eval_results.json"):

    results = []

    metrics = {
        "recall": [],
        "precision": [],
        "mrr": []
    }

    for item in dataset:

        query = item["question"]
        relevant_articles = item.get("relevant_articles", [])
        hard_negatives = item.get("hard_negatives", [])
        hits = rag.rag_service.retriever.retrieve(query, top_k=5)

        retrieved_articles = [
            int(h.payload.get("article_number"))
            for h in hits
            if h.payload.get("article_number") is not None
        ]

        metrics["recall"].append(
            recall_at_k(retrieved_articles, relevant_articles)
        )

        metrics["precision"].append(
            precision_at_k(retrieved_articles, relevant_articles)
        )

        metrics["mrr"].append(
            mrr_at_k(retrieved_articles, relevant_articles)
        )

        reranked = rag.rag_service.reranker.rerank(query, hits)

        contexts = [c.text for c in reranked[:5]]

        response = rag.ask(query)

        results.append({
            "question": query,
            "answer": response.answer,
            "contexts": contexts,
            "relevant_articles": relevant_articles,
            "hard_negatives": hard_negatives,
            "retrieved_articles": retrieved_articles
        })

    report = {
        "recall@5": sum(metrics["recall"]) / len(dataset),
        "precision@5": sum(metrics["precision"]) / len(dataset),
        "mrr@5": sum(metrics["mrr"]) / len(dataset),
    }

    print("\n================ RETRIEVER METRICS ================\n")
    print(report)

    output = {
        "retriever_metrics": report,
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