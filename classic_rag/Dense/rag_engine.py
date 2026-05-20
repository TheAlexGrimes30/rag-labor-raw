from pathlib import Path

from qdrant_client import QdrantClient

from classic_rag.Dense.generator import QwenClient, Generator, LaborPromptBuilder, ContextCleaner
from classic_rag.Dense.index_service import IndexService
from classic_rag.Dense.ingestion import MarkdownDocumentLoader, IngestionPipeline, IngestionService
from classic_rag.Dense.rag_chunkers import HybridLegalChunker
from classic_rag.Dense.rag_config import RAGResponse
from classic_rag.Dense.rag_service import RAGService
from classic_rag.Dense.reranker import Reranker
from classic_rag.Dense.dense_retriever import Embedder, Retriever
from classic_rag.Dense.storage import VectorStore


class RAG:

    def __init__(self):
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        rag_db_path = project_root / "rag_db2"

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
            collection_name="credit_collection",
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


if __name__ == "__main__":

    rag = RAG()

    questions = [
        "какие действия может выполнять должник"
    ]

    try:

        for q in questions:

            print("\n" + "=" * 80)

            print("QUESTION:")
            print(q)

            print("=" * 80)

            res = rag.ask(q)

            print("\nANSWER:")
            print(res.answer)

            print("\nSOURCES:")
            print(res.sources)

    finally:

        rag.close()