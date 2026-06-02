import pandas as pd
from tqdm import tqdm
import json

from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
)

from ragas.dataset_schema import (
    SingleTurnSample,
    EvaluationDataset,
)

from classic_rag.Dense.rag_engine import RAG

# =========================
# LOAD DATASET
# =========================

dataset = pd.read_csv("rag_eval_dataset.csv").iloc[:1]

print("\n[DEBUG] Dataset columns:")
print(dataset.columns)

print("\n[DEBUG] Dataset preview:")
print(dataset.head())

# =========================
# INIT RAG
# =========================

rag = RAG()

samples = []

try:

    # =========================
    # GENERATE ANSWERS
    # =========================

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        query = row["query"]
        true_result = row["result"]

        # ASK RAG
        rag_response = rag.ask(query)

        generated = rag_response.answer

        contexts = getattr(rag_response, "contexts", [])

        print("\n=========================")
        print("QUERY:")
        print(query)

        print("\nTRUE RESULT:")
        print(true_result)

        print("\nGENERATED:")
        print(generated)

        print("\nCONTEXTS:")
        for i, ctx in enumerate(contexts):
            print(f"{i+1}. {str(ctx)[:300]}")

        # BUILD SAMPLE
        sample = SingleTurnSample(
            user_input=query,
            response=generated,
            reference=true_result,
            retrieved_contexts=contexts
        )

        samples.append(sample)

    # =========================
    # SAVE RAW SAMPLES
    # =========================

    raw_samples = []

    for s in samples:
        raw_samples.append({
            "query": s.user_input,
            "true_result": s.reference,
            "generated": s.response,
            "retrieved_contexts": s.retrieved_contexts,
        })

    with open(
        "ragas_samples.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            raw_samples,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n[INFO] Saved ragas_samples.json")

    # =========================
    # CREATE EVAL DATASET
    # =========================

    eval_dataset = EvaluationDataset(samples=samples)

    # =========================
    # RUN RAGAS EVALUATION
    # =========================

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
        ]
    )

    # =========================
    # CONVERT RESULTS
    # =========================

    result_df = result.to_pandas()

    print("\n=== RAGAS RESULT ===")
    print(result_df)

    # =========================
    # BUILD FINAL DATAFRAME
    # =========================

    final_df = pd.DataFrame({
        "query": dataset["query"],
        "true_result": dataset["result"],
        "generated": [s.response for s in samples],
        "faithfulness": result_df["faithfulness"],
        "answer_relevancy": result_df["answer_relevancy"],
        "context_precision": result_df["context_precision"],
    })

    # =========================
    # COMBINED METRICS JSON
    # =========================

    final_df["metric_values"] = final_df.apply(
        lambda row: {
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_precision": row["context_precision"],
        },
        axis=1
    )

    # =========================
    # SAVE CSV
    # =========================

    final_df.to_csv(
        "rag_eval_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[INFO] Saved rag_eval_results.csv")

    # =========================
    # SAVE JSON
    # =========================

    final_df.to_json(
        "rag_eval_results.json",
        orient="records",
        force_ascii=False,
        indent=2
    )

    print("\n[INFO] Saved rag_eval_results.json")

    # =========================
    # FINAL OUTPUT
    # =========================

    print("\n=== FINAL RESULTS ===")
    print(final_df.head())

finally:

    # =========================
    # CLEANUP
    # =========================

    print("\nShutting down RAG...")
    rag.close()