from pathlib import Path
from typing import List

import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient

from classic_rag.Hybrid.generator import Generator, QwenClient, LaborPromptBuilder, ContextCleaner
from classic_rag.Hybrid.ingestion import MarkdownDocumentLoader, IngestionPipeline
from classic_rag.Hybrid.rag_chunkers import SentenceChunker, WindowChunker, StructureChunker, SmartChunker
from classic_rag.Hybrid.rag_config import RAGResponse, SearchResult
from classic_rag.Hybrid.reranker import Reranker
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
    def __init__(self, retriever: Retriever, reranker: Reranker, generator: Generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(query, top_k=10)

        print("\n--- RETRIEVER RESULTS ---")
        for h in hits[:5]:
            print(f"[{h.score:.4f}] ({h.source}) {h.text[:80]}...")

        reranked = self.reranker.rerank(query, hits, top_n=5)

        print("\n--- RERANKED RESULTS ---")
        for h in reranked:
            print(f"[{h.score:.4f}] ({h.source}) {h.text[:80]}...")

        context = self._build_context(reranked)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[h.payload for h in reranked]
        )

    def _build_context(self, hits: List[SearchResult]) -> str:
        parts = []

        for h in hits:
            src = h.payload.get("file", "unknown")
            article = h.payload.get("article_number")
            header = h.payload.get("header")

            if len(h.text) < 100:
                continue

            meta = []

            if article:
                meta.append(f"Статья {article} ТК РФ")

            if header:
                meta.append(header)

            meta_str = " | ".join(meta)

            parts.append(f"[SOURCE: {src} | {meta_str}]\n{h.text}")

        return "\n\n---\n\n".join(parts)[:2500]
    
class ClassicRAG:

    def __init__(self):
        print("Loading documents...")
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        rag_db_path = project_root / "rag_db"

        self.loader = MarkdownDocumentLoader(str(rag_db_path))

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

        self.chunker = SmartChunker(
            structure_chunker=structure_chunker,
            sentence_chunker=sentence_chunker,
            window_chunker=window_chunker,
            max_chars=800
        )

        self.pipeline = IngestionPipeline(
            loader=self.loader,
            chunker=self.chunker
        )

        print("Running ingestion pipeline...")
        self.chunks = self.pipeline.run()

        print(f"Total chunks: {len(self.chunks)}")

        self.embedder = Embedder(
            model_name="intfloat/multilingual-e5-base"
        )

        self.client = QdrantClient(host="localhost", port=6333)

        self.vector_store = VectorStore(
            client=self.client,
            collection_name="rag_collection",
            vector_size=self.embedder.dim
        )

        self.vector_store.ensure_collection()

        print("Indexing chunks into Qdrant...")
        self._index_chunks()

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        self.reranker = Reranker()

        base_dir = Path(__file__).resolve().parents[2]
        model_path = base_dir / "models" / "Qwen3-8B-Q4_K_M.gguf"

        llm = QwenClient(str(model_path))

        self.generator = Generator(
            llm=llm,
            prompt_builder=LaborPromptBuilder(),
            cleaner=ContextCleaner()
        )

    def _index_chunks(self):

        texts = [c.page_content for c in self.chunks]

        payloads = []
        for c in self.chunks:
            p = dict(c.metadata)
            p["text"] = c.page_content
            payloads.append(p)

        vectors = self.embedder.encode_passages(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        self.vector_store.upsert(ids, vectors, payloads)

    def retrieve(self, query: str) -> List[SearchResult]:
        return self.retriever.retrieve(query, top_k=10)

    def rerank(self, query: str, hits: List[SearchResult]) -> List[SearchResult]:
        return self.reranker.rerank(query, hits, top_n=5)

    def build_context(self, hits: List[SearchResult]) -> str:
        parts = []

        for h in hits:
            src = h.payload.get("file", "unknown")
            article = h.payload.get("article_number")
            header = h.payload.get("header")

            if len(h.text) < 100:
                continue

            meta = []

            if article:
                meta.append(f"Статья {article} ТК РФ")

            if header:
                meta.append(header)

            meta_str = " | ".join(meta)

            parts.append(f"[SOURCE: {src} | {meta_str}]\n{h.text}")

        return "\n\n---\n\n".join(parts)[:2500]

    def ask(self, query: str) -> RAGResponse:

        hits = self.retrieve(query)

        print("\n--- RETRIEVER RESULTS ---")
        for h in hits[:5]:
            print(f"[{h.score:.4f}] ({h.source}) {h.text[:80]}...")

        reranked = self.rerank(query, hits)

        print("\n--- RERANKED RESULTS ---")
        for h in reranked:
            print(f"[{h.score:.4f}] ({h.source}) {h.text[:80]}...")

        context = self.build_context(reranked)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[h.payload for h in reranked]
        )

    def close(self):
        print("Shutting down RAG...")

        if hasattr(self.generator, "llm") and hasattr(self.generator.llm, "close"):
            self.generator.llm.close()

        self.generator = None
        self.retriever = None
        self.reranker = None

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
