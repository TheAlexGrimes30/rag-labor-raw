from __future__ import annotations

from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from classic_rag.Dense.dense_retriever import Embedder, Retriever
from classic_rag.Dense.generator import QwenClient, LaborPromptBuilder, Generator, ContextCleaner
from classic_rag.Dense.index_service import IndexService
from classic_rag.Dense.ingestion import IngestionPipeline, MarkdownDocumentLoader, IngestionService
from classic_rag.Dense.rag_chunkers import HybridLegalChunker
from classic_rag.Dense.rag_dataset import dataset
from classic_rag.Dense.rag_service import RAGService
from classic_rag.Dense.reranker import Reranker
from classic_rag.Dense.retrieval_evaluation import evaluate_rag
from classic_rag.Dense.search_result import SearchResult
from classic_rag.Dense.storage import VectorStore


class RAG:
    """
    Main RAG pipeline.

    Responsibilities:
    - document ingestion
    - chunk indexing
    - vector retrieval
    - reranking
    - answer generation
    """

    def __init__(self) -> None:
        """
        Initialize all RAG components.

        Returns:
            None
        """

        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        model_path = project_root / "models" / "Qwen3-8B-Q4_K_M.gguf"

        rag_db_path = project_root / "rag_db"

        self.debug_path = project_root / "debug"



        loader = MarkdownDocumentLoader(
            str(rag_db_path)
        )

        parser = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=parser
        )

        self.ingestion = IngestionService(
            pipeline
        )


        self.embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            normalize=True
        )


        self.qdrant = QdrantClient(
            "localhost",
            port=6333
        )

        self.vector_store = VectorStore(
            client=self.qdrant,
            collection_name="labor_dense_collection_6",
            vector_size=self.embedder.dim,
            distance=Distance.COSINE
        )

        self.vector_store.ensure_collection()


        self.index_service = IndexService(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            max_pool_size=50
        )


        self.reranker = Reranker(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=5
        )

        self.llm = QwenClient(model_path=str(model_path))

        self.prompt_builder = LaborPromptBuilder()

        self.cleaner = ContextCleaner()

        self.generator = Generator(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            cleaner=self.cleaner
        )

        self.rag_service = RAGService(
            retriever=self.retriever,
            reranker=self.reranker,
            generator=self.generator,
            max_context_chars=3500,
            min_final_score=0.50
        )

    def build_and_index(self) -> List:
        """
        Load chunks and index them if collection is empty.

        Returns:
            List:
                List of loaded chunks.
        """

        print("\nLoading chunks...\n")

        chunks = self.ingestion.load_chunks()

        print(f"Loaded chunks: {len(chunks)}")

        self.index_if_needed(chunks)

        return chunks

    def index_if_needed(
        self,
        chunks: List
    ) -> None:
        """
        Index chunks only if collection is empty.

        Args:
            chunks (List):
                List of document chunks.

        Returns:
            None
        """

        collection = self.vector_store.collection_name

        info = self.qdrant.get_collection(
            collection
        )

        points_count = info.points_count

        if points_count > 0:

            print(
                f"[Index] Skipping indexing — "
                f"collection already has "
                f"{points_count} points"
            )

            return

        print(
            "[Index] Collection empty — "
            "starting indexing..."
        )

        self.index_service.index(chunks)

        print("[Index] Done indexing")

    def search(
        self,
        query: str,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 5,
        use_reranker: bool = True
    ) -> List[SearchResult]:
        """
        Perform retrieval and optional reranking.

        Args:
            query (str):
                User query.

            retrieve_top_k (int):
                Number of retrieved chunks.

            rerank_top_n (int):
                Number of final reranked chunks.

            use_reranker (bool):
                Whether to apply reranking.

        Returns:
            List[SearchResult]:
                Final search results.
        """

        hits = self.retriever.retrieve(
            query=query,
            top_k=retrieve_top_k
        )

        if not use_reranker:
            return hits[:rerank_top_n]

        return self.reranker.rerank(
            query=query,
            hits=hits,
            top_n=rerank_top_n
        )

    def ask(
        self,
        query: str
    ) -> str:
        """
        Generate final answer using full RAG pipeline.

        Args:
            query (str):
                User query.

        Returns:
            str:
                Final generated answer.
        """

        response = self.rag_service.ask(
            query=query
        )

        return response.answer


    def debug_dense_retrieval(
        self,
        query: str,
        top_k: int = 10
    ) -> None:
        """
        Debug dense retrieval results.

        Args:
            query (str):
                Search query.

            top_k (int):
                Number of retrieved chunks.

        Returns:
            None
        """

        print("\n" + "=" * 100)

        print("[DENSE RETRIEVAL DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        hits = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )

        for i, hit in enumerate(
            hits,
            start=1
        ):

            payload = hit.payload or {}

            print("\n" + "-" * 100)

            print(f"DENSE TOP {i}")

            print(
                f"SCORE   : "
                f"{hit.score:.4f}"
            )

            print(
                f"ARTICLE : "
                f"{payload.get('article_number', 'unknown')}"
            )

            print(
                f"HEADER  : "
                f"{payload.get('header', 'unknown')}"
            )

            print("\nTEXT:\n")

            print(
                (hit.text or "")[:1200]
            )


    def debug_search_pipeline(
        self,
        query: str,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 5
    ) -> None:
        """
        Debug full retrieval + reranking pipeline.

        Args:
            query (str):
                User query.

            retrieve_top_k (int):
                Retriever top-k.

            rerank_top_n (int):
                Final reranker top-n.

        Returns:
            None
        """

        print("\n" + "=" * 100)

        print("[FULL SEARCH PIPELINE DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        dense_hits = self.retriever.retrieve(
            query=query,
            top_k=retrieve_top_k
        )

        print("\n" + "=" * 100)

        print("RERANKER OUTPUT")

        print("=" * 100)

        final_hits = self.reranker.rerank(
            query=query,
            hits=dense_hits,
            top_n=rerank_top_n
        )

        for i, hit in enumerate(
            final_hits,
            start=1
        ):

            payload = hit.payload or {}

            print("\n" + "-" * 100)

            print(f"FINAL TOP {i}")

            print(
                f"SCORE   : "
                f"{hit.score:.4f}"
            )

            print(
                f"ARTICLE : "
                f"{payload.get('article_number')}"
            )

            print(
                f"HEADER  : "
                f"{payload.get('header')}"
            )

            print("\nTEXT:\n")

            print(
                (hit.text or "")[:1200]
            )

    def close(self) -> None:
        """
        Release resources.
        """

        try:
            if self.generator:
                self.generator.close()
        except Exception:
            pass

        try:
            if self.qdrant:
                self.qdrant.close()
        except Exception:
            pass

        try:
            if self.embedder:
                del self.embedder
        except Exception:
            pass

        try:
            if self.generator:
                self.generator.close()
        except Exception:
            pass

        print("\nShutting down...")


if __name__ == "__main__":

    rag = RAG()

    try:

        rag.build_and_index()

        print("\nIndex ready.\n")


        #query = (
        #    "Можно ли работать с 14 лет?"
        #)

        #print("\n" + "=" * 100)

        #print("[GENERATOR TEST]")

        #print("=" * 100)

        #print(f"\nQUERY:\n{query}")

        #print("\n" + "=" * 100)

        #answer = rag.ask(query)

        #print("\nANSWER:\n")

        #print(answer)

        print("\n" + "#" * 100)
        print("[RAG EVALUATION START]")
        print("#" * 100)

        evaluate_rag(
            rag,
            dataset,
            output_path="rag_eval_results_6.json",
            use_reranker=True,
            retrieve_top_k=20,
            rerank_top_n=5
        )

    finally:

        rag.close()