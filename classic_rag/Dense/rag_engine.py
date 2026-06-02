from __future__ import annotations

from pathlib import Path
from typing import Any, List

from llama_index.core import Document, PropertyGraphIndex
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from classic_rag.Dense.dense_retriever import Embedder
from classic_rag.Dense.generator import (
    ContextCleaner,
    Generator,
    LaborPromptBuilder,
    QwenClient,
)
from classic_rag.Dense.index_service import IndexService
from classic_rag.Dense.ingestion import (
    IngestionPipeline,
    IngestionService,
    MarkdownDocumentLoader,
)
from classic_rag.Dense.rag_chunkers import HybridLegalChunker
from classic_rag.Dense.rag_dataset import dataset
from classic_rag.Dense.rag_service import RAGService
from classic_rag.Dense.reranker import Reranker
from classic_rag.Dense.retrieval_evaluation import evaluate_rag
from classic_rag.Dense.search_result import SearchResult
from classic_rag.Dense.storage import VectorStore

from classic_rag.Hybrid.hybrid_retriever import (
    BaseGraphRetriever,
    LlamaIndexGraphRetriever,
    Retriever,
)


class EmptyGraphRetriever(BaseGraphRetriever):
    """
    Temporary graph retriever stub.

    Used before chunks are loaded and the real LlamaIndex graph index
    is built.
    """

    def search(
            self,
            query: str,
            k: int
    ) -> list[SearchResult]:
        """
        Return empty graph results.
        """

        return []


class RAG:
    """
    Main RAG pipeline.

    Responsibilities:
    - document ingestion
    - dense vector indexing
    - graph index building
    - hybrid retrieval
    - reranking
    - answer generation
    """

    def __init__(self) -> None:
        """
        Initialize all RAG components.
        """

        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]

        llm_path = (
            project_root
            / "models"
            / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
        )

        embedder_path = (
            project_root
            / "models"
            / "Qwen3-Embedding-0.6B"
        )

        reranker_path = (
            project_root
            / "models"
            / "bge-reranker-v2-m3"
        )

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
            model_name=str(embedder_path),
            normalize=True
        )

        self.qdrant = QdrantClient(
            "localhost",
            port=6333
        )

        self.vector_store = VectorStore(
            client=self.qdrant,
            collection_name="labor_dense_collection",
            vector_size=self.embedder.dim,
            distance=Distance.COSINE
        )

        self.vector_store.ensure_collection()

        self.index_service = IndexService(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        self.graph_retriever: BaseGraphRetriever = (
            EmptyGraphRetriever()
        )

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            graph_retriever=self.graph_retriever
        )

        self.reranker = Reranker(
            model_name=str(reranker_path),
            top_n=5
        )

        self.llm = QwenClient(
            model_path=str(llm_path)
        )

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

    def build_and_index(self) -> List[Any]:
        """
        Load chunks, index them in Qdrant if needed,
        and build LlamaIndex Graph RAG retriever.
        """

        print("\nLoading chunks...\n")

        chunks = self.ingestion.load_chunks()

        print(f"Loaded chunks: {len(chunks)}")

        self.index_if_needed(chunks)

        print("\nBuilding Graph RAG index...\n")

        self.graph_retriever = self._build_graph_retriever(
            chunks
        )

        self.retriever.graph = self.graph_retriever

        print("\nGraph RAG index ready.\n")

        return chunks

    def index_if_needed(
            self,
            chunks: List[Any]
    ) -> None:
        """
        Index chunks only if Qdrant collection is empty.
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

    def _metadata_to_dict(
            self,
            metadata: Any
    ) -> dict[str, Any]:
        """
        Convert chunk metadata object to plain dict.

        Supports:
        - dict
        - pydantic v2 model_dump()
        - pydantic v1 dict()
        - regular Python objects with __dict__
        """

        if metadata is None:
            return {}

        if isinstance(metadata, dict):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(
                metadata.model_dump()
            )

        if hasattr(metadata, "dict"):
            return dict(
                metadata.dict()
            )

        if hasattr(metadata, "__dict__"):
            return {
                key: value
                for key, value in metadata.__dict__.items()
                if not key.startswith("_")
            }

        return {}

    def _extract_chunk_text(
            self,
            chunk: Any
    ) -> str:
        """
        Extract text from different chunk object shapes.
        """

        text = (
            getattr(chunk, "text", None)
            or getattr(chunk, "content", None)
            or getattr(chunk, "page_content", None)
            or ""
        )

        return str(text)

    def _extract_chunk_metadata(
            self,
            chunk: Any
    ) -> dict[str, Any]:
        """
        Extract and normalize metadata from chunk.
        """

        metadata_raw = (
            getattr(chunk, "metadata", None)
            or getattr(chunk, "payload", None)
            or {}
        )

        metadata = self._metadata_to_dict(
            metadata_raw
        )

        article_number = metadata.get(
            "article_number"
        )

        if article_number is None:
            article_number = metadata.get(
                "article"
            )

        if article_number is not None:
            metadata["article_number"] = str(
                article_number
            ).strip()

        header = metadata.get("header")

        if header is not None:
            metadata["header"] = str(
                header
            ).strip()

        return metadata

    def _build_graph_retriever(
            self,
            chunks: List[Any]
    ) -> BaseGraphRetriever:
        """
        Build LlamaIndex PropertyGraphIndex retriever from loaded chunks.
        """

        documents: list[Document] = []

        for chunk in chunks:
            text = self._extract_chunk_text(
                chunk
            )

            metadata = self._extract_chunk_metadata(
                chunk
            )

            if not text.strip():
                continue

            documents.append(
                Document(
                    text=text,
                    metadata=metadata
                )
            )

        if not documents:
            print(
                "[Graph RAG] No documents found "
                "for graph index."
            )

            return EmptyGraphRetriever()

        graph_index = PropertyGraphIndex.from_documents(
            documents,
            show_progress=True
        )

        llama_retriever = graph_index.as_retriever(
            similarity_top_k=30
        )

        return LlamaIndexGraphRetriever(
            graph_retriever=llama_retriever
        )

    def search(
            self,
            query: str,
            retrieve_top_k: int = 20,
            rerank_top_n: int = 5,
            use_reranker: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid retrieval and optional reranking.
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
        Generate final answer using the full RAG pipeline.
        """

        response = self.rag_service.ask(
            query=query
        )

        return response.answer

    def debug_hybrid_retrieval(
            self,
            query: str,
            top_k: int = 10
    ) -> None:
        """
        Debug hybrid dense + graph retrieval results.
        """

        self.retriever.debug_query(
            query=query,
            top_k=top_k
        )

    def debug_search_pipeline(
            self,
            query: str,
            retrieve_top_k: int = 20,
            rerank_top_n: int = 5
    ) -> None:
        """
        Debug full hybrid retrieval + reranking pipeline.
        """

        print("\n" + "=" * 100)
        print("[FULL HYBRID SEARCH PIPELINE DEBUG]")
        print(f"QUERY: {query}")
        print("=" * 100)

        retrieved_hits = self.retriever.retrieve(
            query=query,
            top_k=retrieve_top_k
        )

        print("\n" + "=" * 100)
        print("RERANKER OUTPUT")
        print("=" * 100)

        final_hits = self.reranker.rerank(
            query=query,
            hits=retrieved_hits,
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

            print(
                f"SOURCE  : "
                f"{payload.get('retrieval_source') or payload.get('retrieval_sources')}"
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

        print("\nShutting down...")


if __name__ == "__main__":

    rag = RAG()

    try:
        rag.build_and_index()

        print("\nIndex ready.\n")

        print("\n" + "#" * 100)
        print("[RAG EVALUATION START]")
        print("#" * 100)

        evaluate_rag(
            rag,
            dataset,
            output_path="rag_eval_results_hybrid_graph.json",
            use_reranker=True,
            retrieve_top_k=20,
            rerank_top_n=5
        )

    finally:
        rag.close()