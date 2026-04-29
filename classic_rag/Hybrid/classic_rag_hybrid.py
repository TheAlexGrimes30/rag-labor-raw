from pathlib import Path
from typing import List

from qdrant_client import QdrantClient

from classic_rag.Hybrid.generator import Generator, QwenClient, LaborPromptBuilder, ContextCleaner
from classic_rag.Hybrid.ingestion import MarkdownDocumentLoader, IngestionPipeline
from classic_rag.Hybrid.rag_chunkers import SentenceChunker, WindowChunker, StructureChunker, SmartChunker
from classic_rag.Hybrid.rag_config import RAGResponse, SearchResult, Chunk
from classic_rag.Hybrid.retriever import Retriever, Embedder
from classic_rag.Hybrid.storage import VectorStore

class IngestionService:
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def load_chunks(self) -> List[Chunk]:
        chunks = self.pipeline.run()
        print(f"[Ingestion] Loaded chunks: {len(chunks)}")
        return chunks

class IndexService:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        payloads = [c.to_payload() for c in chunks]
        ids = [c.chunk_id for c in chunks]

        vectors = self.embedder.encode_passages(texts)

        self.vector_store.upsert(ids, vectors, payloads)

        print(f"[Index] Indexed: {len(chunks)} chunks")

class RAGService:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str) -> RAGResponse:

        hits = self.retriever.retrieve(query, top_k=15)

        print("\n--- RETRIEVER RESULTS ---")
        for h in hits[:7]:
            print(f"[{h.score:.4f}] ({h.source}) {h.text[:80]}...")

        top_hits = self._select_hits(hits, query)

        context = self._build_context(top_hits)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[h.payload for h in top_hits]
        )

    def _select_hits(self, hits: List[SearchResult], query: str) -> List[SearchResult]:

        selected = []
        seen = set()

        query_lower = query.lower()
        query_words = query_lower.split()

        for h in hits:
            text = (h.text or "").strip()

            if len(text) < 40:
                continue

            key = hash(text)
            if key in seen:
                continue

            seen.add(key)

            header = (h.payload.get("header") or "").lower()

            if h.payload.get("article_number"):
                selected.insert(0, h)
                continue

            if any(word in header for word in query_words):
                selected.insert(0, h)
                continue

            if any(word in text.lower() for word in query_words):
                selected.append(h)
            else:
                selected.append(h)

            if len(selected) >= 8:
                break

        return selected

    def _build_context(self, hits: List[SearchResult]) -> str:
        parts = []

        for i, h in enumerate(hits[:6]):
            text = (h.text or "").strip()

            if len(text) < 40:
                continue

            article = h.payload.get("article_number")
            header = h.payload.get("header")

            meta = []

            if article:
                meta.append(f"Статья {article} ТК РФ")

            if header:
                meta.append(header)

            meta_str = " | ".join(meta)

            parts.append(f"""
    ФРАГМЕНТ {i + 1}
    {meta_str}
    {text}
    """.strip())

        return "\n\n".join(parts)

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

        print(f"\n[DEBUG] Total chunks: {len(chunks)}")

        for i, c in enumerate(chunks[:5]):
            payload = c.to_payload()

            print(f"\n--- CHUNK {i} ---")
            print(f"text: {c.text[:200]}")
            print(f"file: {payload.get('file')}")
            print(f"article: {payload.get('article_number')}")
            print(f"header: {payload.get('header')}")

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
