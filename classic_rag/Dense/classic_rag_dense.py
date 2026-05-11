import json
from pathlib import Path
from typing import List

import pandas as pd
from qdrant_client import QdrantClient

from classic_rag.Dense.generator import Generator, QwenClient, LaborPromptBuilder, ContextCleaner
from classic_rag.Dense.index_service import IndexService
from classic_rag.Dense.ingestion import IngestionPipeline, MarkdownDocumentLoader, IngestionService
from classic_rag.Dense.rag_chunkers import HybridLegalChunker
from classic_rag.Dense.rag_config import RAGResponse, Chunk
from classic_rag.Dense.rag_service import RAGService
from classic_rag.Dense.reranker import Reranker
from classic_rag.Dense.retriever import Retriever, Embedder
from classic_rag.Dense.storage import VectorStore

dataset = [

    {
        "id": 1,
        "question": "какие цели трудового законодательства",
        "relevant_articles": [1],
        "hard_negatives": [2, 11, 56],
        "category": "general_definition"
    },

    {
        "id": 2,
        "question": "что регулирует трудовое законодательство",
        "relevant_articles": [1, 11],
        "hard_negatives": [10, 56, 106],
        "category": "scope"
    },

    {
        "id": 3,
        "question": "обязанности работодателя",
        "relevant_articles": [22, 15],
        "hard_negatives": [56, 91],
        "category": "employer_rights"
    },

    {
        "id": 4,
        "question": "что такое трудовой договор",
        "relevant_articles": [56],
        "hard_negatives": [15, 22],
        "category": "definitions"
    },

    {
        "id": 5,
        "question": "время отдыха работников",
        "relevant_articles": [106, 107, 108],
        "hard_negatives": [1, 10],
        "category": "labor_conditions"
    },

    {
        "id": 6,
        "question": "перерывы в рабочем времени",
        "relevant_articles": [108],
        "hard_negatives": [106, 107],
        "category": "working_time"
    },

    {
        "id": 7,
        "question": "международные нормы в трудовом праве",
        "relevant_articles": [10],
        "hard_negatives": [1, 11],
        "category": "legal_system"
    },

    {
        "id": 8,
        "question": "контроль за соблюдением трудового законодательства",
        "relevant_articles": [1, 11],
        "hard_negatives": [22, 56],
        "category": "supervision"
    }
]

class ClassicRAG:

    def __init__(self):
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        rag_db_path = project_root / "rag_db"

        loader = MarkdownDocumentLoader(str(rag_db_path))

        parser = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=parser
        )

        self.ingestion = IngestionService(pipeline)

        embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        client = QdrantClient(host="localhost", port=6333)

        vector_store = VectorStore(
            client=client,
            collection_name="labor_rag_dense_collection",
            vector_size=embedder.dim
        )

        vector_store.ensure_collection()

        self.index_service = IndexService(vector_store, embedder)

        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder
        )


        model_path = project_root / "models" / "Qwen3-8B-Q4_K_M.gguf"

        llm = QwenClient(str(model_path))

        generator = Generator(
            llm=llm,
            prompt_builder=LaborPromptBuilder(),
            cleaner=ContextCleaner()
        )

        reranker = Reranker()
        self.rag_service = RAGService(retriever, reranker, generator)
        self.client = client
        self.generator = generator

        print("Running ingestion...")
        self.chunks = self.ingestion.load_chunks()

        print(f"\n[DEBUG] Total chunks: {len(self.chunks)}")

        for c in self.chunks[:20]:
            print(c.metadata.article_number, "|", c.metadata.header)

        for i, c in enumerate(self.chunks[:5]):
            payload = c.to_payload()

            print(f"\n--- CHUNK {i} ---")
            print(f"text: {c.text[:200]}")
            print(f"file: {payload.get('file')}")
            print(f"article: {payload.get('article_number')}")
            print(f"header: {payload.get('header')}")

        print("Indexing...")
        self.index_service.index(self.chunks)


    def ask(self, query: str) -> RAGResponse:
        return self.rag_service.ask(query)

    def close(self):
        print("Shutting down RAG...")

        try:
            if hasattr(self.generator, "llm"):
                self.generator.llm = None

        except Exception as e:
            print("[WARN] LLM cleanup error:", repr(e))

        try:
            self.client.close()
        except Exception as e:
            print("[WARN] Qdrant close error:", repr(e))



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

if __name__ == "__main__":

    rag = ClassicRAG()
    retriever = rag.rag_service.retriever
    reranker = rag.rag_service.reranker
    chunks = rag.chunks

    try:
        evaluate_rag(rag, dataset)


    except Exception as e:
        print("\n[CRITICAL ERROR] Сбой всей RAG системы:")
        print(repr(e))

    finally:
        try:
            rag.close()
        except Exception as e:
            print("\n[WARN] Ошибка при закрытии ресурсов:", repr(e))