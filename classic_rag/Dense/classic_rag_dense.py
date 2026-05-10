from pathlib import Path
from typing import List

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

def debug_chunks(chunks: List[Chunk]):
    print("\n[DEBUG] ===== CHUNK QUALITY =====")

    total = len(chunks)
    short = 0
    duplicates = 0

    seen = set()

    for c in chunks:
        text = (c.text or "").strip()

        if len(text) < 50:
            short += 1

        key = text[:200]
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    print(f"Total chunks: {total}")
    print(f"Short chunks (<50 chars): {short}")
    print(f"Duplicates: {duplicates}")
    print(f"Unique chunks: {total - duplicates}")

def save_chunks_to_txt(chunks, path="debug_chunks.txt"):
    with open(path, "w", encoding="utf-8") as f:

        for i, c in enumerate(chunks):

            payload = c.to_payload()

            f.write(f"\n--- CHUNK {i} ---\n")
            f.write(f"text:\n{c.text}\n\n")
            f.write(f"file: {payload.get('file')}\n")
            f.write(f"article: {payload.get('article_number')}\n")
            f.write(f"header: {payload.get('header')}\n")
            f.write(f"level: {payload.get('level')}\n")
            f.write(f"topics: {payload.get('topics')}\n")
            f.write("-" * 60 + "\n")

if __name__ == "__main__":

    rag = ClassicRAG()
    retriever = rag.rag_service.retriever
    reranker = rag.rag_service.reranker
    chunks = rag.chunks

    try:
        questions = [
            "какие цели трудового законодательства"
        ]

        for q in questions:
            print("\nQ:", q)

            try:
                retriever.debug_query(q, top_k=10)

                hits = retriever.retrieve(q, top_k=25)

                reranker.debug_rerank(q, hits)

                res = rag.ask(q)

                print("A:", res.answer)

            except Exception as e:
                print("\n[ERROR] Ошибка при обработке вопроса:")
                print("Question:", q)
                print("Error:", repr(e))

                continue

    except Exception as e:
        print("\n[CRITICAL ERROR] Сбой всей RAG системы:")
        print(repr(e))

    finally:
        try:
            rag.close()
        except Exception as e:
            print("\n[WARN] Ошибка при закрытии ресурсов:", repr(e))