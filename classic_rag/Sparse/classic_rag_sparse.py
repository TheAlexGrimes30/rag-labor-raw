from pathlib import Path
from typing import List

from classic_rag.Sparse.generator import ContextCleaner, LaborPromptBuilder, QwenClient, Generator
from classic_rag.Sparse.index_service import IndexService
from classic_rag.Sparse.ingestion import IngestionService, MarkdownDocumentLoader, IngestionPipeline
from classic_rag.Sparse.rag_chunkers import HybridLegalChunker
from classic_rag.Sparse.rag_config import RAGResponse, Chunk
from classic_rag.Sparse.rag_service import RAGService
from classic_rag.Sparse.reranker import Reranker
from classic_rag.Sparse.retriever import BM25Retriever


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

        model_path = project_root / "models" / "Qwen3-8B-Q4_K_M.gguf"

        llm = QwenClient(str(model_path))

        generator = Generator(
            llm=llm,
            prompt_builder=LaborPromptBuilder(),
            cleaner=ContextCleaner()
        )


        self.index_service = IndexService()

        self.chunks = self.ingestion.load_chunks()

        print("Indexing BM25...")
        self.index_service.index(self.chunks)

        self.retriever: BM25Retriever = self.index_service.get_retriever()


        reranker = Reranker()

        self.rag_service = RAGService(
            retriever=self.retriever,
            reranker=reranker,
            generator=generator
        )

        self.generator = generator

        print(f"\n[DEBUG] Total chunks: {len(self.chunks)}")

        for i, c in enumerate(self.chunks[:5]):
            payload = c.to_payload()

            print(f"\n--- CHUNK {i} ---")
            print(f"text: {c.text[:200]}")
            print(f"file: {payload.get('file')}")
            print(f"article: {payload.get('article_number')}")
            print(f"header: {payload.get('header')}")

    def ask(self, query: str) -> RAGResponse:
        return self.rag_service.ask(query)


    def close(self):

        print("Shutting down RAG...")

        try:
            if hasattr(self.generator, "llm"):
                self.generator.llm = None
        except Exception as e:
            print("[WARN] LLM cleanup error:", repr(e))


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

    except Exception as e:
        print("\n[CRITICAL ERROR] Сбой всей RAG системы:")
        print(repr(e))

    finally:
        try:
            rag.close()
        except Exception as e:
            print("\n[WARN] Ошибка при закрытии ресурсов:", repr(e))