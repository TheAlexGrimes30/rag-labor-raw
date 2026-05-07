import hashlib
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient

from classic_rag.Hybrid.generator import Generator, QwenClient, LaborPromptBuilder, ContextCleaner
from classic_rag.Hybrid.ingestion import IngestionPipeline, MarkdownDocumentLoader, IngestionService
from classic_rag.Hybrid.rag_chunkers import HybridLegalChunker
from classic_rag.Hybrid.rag_config import RAGResponse, SearchResult, Chunk
from classic_rag.Hybrid.reranker import Reranker
from classic_rag.Hybrid.retriever import Retriever, Embedder
from classic_rag.Hybrid.storage import VectorStore

class IndexService:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks if c.text]
        payloads = [c.to_payload() for c in chunks if c.text]
        ids = [c.chunk_id for c in chunks if c.text]

        if not texts:
            return

        vectors = self.embedder.encode_passages(texts)

        self.vector_store.upsert(ids, vectors, payloads)

        print(f"[Index] Indexed: {len(texts)} chunks")

class RAGService:

    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def ask(self, query: str) -> RAGResponse:

        # 1. retrieval
        hits = self.retriever.retrieve(query, k=25)

        if not hits:
            return RAGResponse(
                answer="Нет релевантных документов",
                sources=[]
            )

        # 2. deduplicate
        hits = self._deduplicate(hits)

        if not hits:
            return RAGResponse(
                answer="Нет релевантных документов после фильтрации",
                sources=[]
            )

        # 3. rerank
        reranked = self.reranker.rerank(
            query=query,
            hits=hits,
            top_n=6
        )

        if not reranked:
            return RAGResponse(
                answer="Не удалось переоценить документы",
                sources=[]
            )

        # 4. context
        context = self._build_context(reranked)

        if not context.strip():
            return RAGResponse(
                answer="Недостаточно контекста для ответа",
                sources=[]
            )

        # 5. generate
        answer = self.generator.generate(query, context)

        # 6. sources (с сохранением порядка)
        sources = []
        seen = set()

        for h in reranked:
            art = h.payload.get("article_number")
            if art and art not in seen:
                seen.add(art)
                sources.append(f"Статья {art}")

        return RAGResponse(
            answer=answer,
            sources=sources
        )

    def _build_context(self, hits: List[SearchResult]) -> str:

        grouped = {}

        for h in hits:
            article = h.payload.get("article_number") or "unknown"
            grouped.setdefault(article, []).append(h)

        parts = []

        for article, items in grouped.items():

            items = sorted(
                items,
                key=lambda x: float(getattr(x, "score") or 0.0),
                reverse=True
            )

            block = [f"СТАТЬЯ {article}"]

            for h in items:
                text = (h.text or "").strip()
                header = h.payload.get("header") or ""

                if len(text) < 50:
                    continue

                block.append(f"{header}\n{text}")

            parts.append("\n\n".join(block))

        return "\n\n---\n\n".join(parts)

    def _deduplicate(self, hits: List[SearchResult]) -> List[SearchResult]:

        seen = set()
        result = []

        for h in hits:

            text = (h.text or "").strip()

            if len(text) < 50:
                continue

            # 🔥 stable hash вместо slice
            key = hashlib.md5(text.encode("utf-8")).hexdigest()

            if key in seen:
                continue

            seen.add(key)
            result.append(h)

        return result

class ClassicRAG:

    def __init__(self):
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        rag_db_path = project_root / "rag_db"

        # ingestion
        loader = MarkdownDocumentLoader(str(rag_db_path))
        parser = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=parser
        )

        self.ingestion = IngestionService(pipeline)

        # embedder
        self.embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        # vector DB
        self.client = QdrantClient(host="localhost", port=6333)

        self.vector_store = VectorStore(
            client=self.client,
            collection_name="rag_qwen_collection",
            vector_size=self.embedder.dim
        )

        self.vector_store.ensure_collection()

        # services
        self.index_service = IndexService(self.vector_store, self.embedder)

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        llm = QwenClient(str(project_root / "models" / "Qwen3-8B-Q4_K_M.gguf"))

        self.generator = Generator(
            llm=llm,
            prompt_builder=LaborPromptBuilder(),
            cleaner=ContextCleaner()
        )

        self.reranker = Reranker()

        self.rag_service = RAGService(
            self.retriever,
            self.reranker,
            self.generator
        )

        # ❌ НЕ делаем ingestion в __init__
        self.chunks = []

    def build(self):
        """Отдельный этап инициализации данных"""

        print("Running ingestion...")

        self.chunks = self.ingestion.load_chunks()

        print(f"[DEBUG] Total chunks: {len(self.chunks)}")

        self.index_service.index(self.chunks)

        print("Indexing complete.")

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
        rag.build()

        questions = [
            "какие цели трудового законодательства",
            "что регулирует трудовое законодательство",
            "что такое свобода труда",
            "какие принципы трудового права",
        ]

        for q in questions:
            print("\nQ:", q)

            # ❌ убрали debug_query (он дублирует retrieval)
            res = rag.ask(q)

            print("A:", res.answer)

    finally:
        rag.close()