from pathlib import Path
from typing import List

import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from classic_rag.Hybrid.generator import Generator, QwenClient, LaborPromptBuilder, ContextCleaner
from classic_rag.Hybrid.ingestion import MarkdownDocumentLoader, IngestionPipeline
from classic_rag.Hybrid.rag_chunkers import SentenceChunker, WindowChunker, StructureChunker, SmartChunker
from classic_rag.Hybrid.rag_config import RAGResponse, SearchResult
from classic_rag.Hybrid.retriever import Retriever, Embedder
from classic_rag.Hybrid.storage import VectorStore

class IngestionService:
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def load_chunks(self) -> List[Document]:
        chunks = self.pipeline.run()
        print(f"[Ingestion] Loaded chunks: {len(chunks)}")
        return chunks

class IndexService:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def index(self, chunks: List[Document]):
        if not chunks:
            return

        texts = [c.page_content for c in chunks]

        payloads = []
        for c in chunks:
            p = dict(c.metadata)
            p["text"] = c.page_content
            payloads.append(p)

        vectors = self.embedder.encode_passages(texts)
        ids = [str(uuid.uuid4()) for _ in texts]

        self.vector_store.upsert(ids, vectors, payloads)

        print(f"[Index] Indexed: {len(chunks)} chunks")

class RAGService:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(query, top_k=10)

        print("\n--- RETRIEVER RESULTS ---")
        for h in hits[:5]:
            print(f"[{h.score:.4f}] ({h.source}) {h.text[:80]}...")

        top_hits = hits[:5]

        context = self._build_context(top_hits)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[h.payload for h in top_hits]
        )

    def _build_context(self, hits: List[SearchResult]) -> str:
        parts = []

        for h in hits:
            text = (h.text or "").strip()

            if len(text) < 30:
                continue

            src = h.payload.get("file", "unknown")
            article = h.payload.get("article_number")
            header = h.payload.get("header")

            meta = []

            if article:
                meta.append(f"Статья {article} ТК РФ")

            if header:
                meta.append(header)

            meta_str = " | ".join(meta)

            parts.append(f"[SOURCE: {src} | {meta_str}]\n{text}")

        return "\n\n---\n\n".join(parts)[:5000]

class ClassicRAG:

    def __init__(self):
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        rag_db_path = project_root / "rag_db"

        loader = MarkdownDocumentLoader(str(rag_db_path))

        sentence_chunker = SentenceChunker(
            chunk_size=800,
            overlap_sentences=2
        )

        window_chunker = WindowChunker(
            max_chars=800,
            overlap=150
        )

        structure_chunker = StructureChunker(
            fallback_chunker=sentence_chunker
        )

        chunker = SmartChunker(
            structure_chunker=structure_chunker,
            sentence_chunker=sentence_chunker,
            window_chunker=window_chunker,
            max_chars=800
        )

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker
        )

        self.ingestion = IngestionService(pipeline)

        embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        client = QdrantClient(host="localhost", port=6333)

        vector_store = VectorStore(
            client=client,
            collection_name="rag_qwen_collection",
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

        self.rag_service = RAGService(retriever, generator)
        self.client = client
        self.generator = generator

        print("Running ingestion...")
        chunks = self.ingestion.load_chunks()

        print("Indexing...")
        self.index_service.index(chunks)


    def ask(self, query: str) -> RAGResponse:
        return self.rag_service.ask(query)

    def close(self):
        print("Shutting down RAG...")

        if hasattr(self.generator, "llm") and hasattr(self.generator.llm, "close"):
            self.generator.llm.close()

        self.client.close()


if __name__ == "__main__":

    rag = ClassicRAG()

    try:
        questions = [
            "какие цели трудового законодательства",
            "что регулирует трудовое законодательство",
            "что такое свобода труда",
            "какие принципы трудового права",
        ]

        for q in questions:
            print("\nQ:", q)
            res = rag.ask(q)
            print("A:", res.answer)

    finally:
        rag.close()
