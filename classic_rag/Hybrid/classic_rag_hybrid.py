import re
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from classic_rag.Hybrid.generator import Generator, QwenClient, LaborPromptBuilder, ContextCleaner
from classic_rag.Hybrid.rag_config import RAGResponse, SearchResult
from classic_rag.Hybrid.reranker import Reranker
from classic_rag.Hybrid.retriever import Retriever, Embedder
from classic_rag.Hybrid.storage import VectorStore


class ClassicRAG:

    def __init__(self):
        print("Loading documents...")
        self.documents = self.load_documents()

        print("Chunking...")
        self.chunks = self.chunk_documents(self.documents)

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
        prompt_builder = LaborPromptBuilder()
        cleaner = ContextCleaner()

        self.generator = Generator(
            llm=llm,
            prompt_builder=prompt_builder,
            cleaner=cleaner
        )

    def _index_chunks(self):

        texts = [c.page_content for c in self.chunks]

        payloads = []
        for c in self.chunks:
            p = dict(c.metadata)
            p["text"] = c.page_content
            payloads.append(p)

        vectors = self.embedder.encode_passages(texts)

        import uuid
        ids = [str(uuid.uuid4()) for _ in texts]

        self.vector_store.upsert(ids, vectors, payloads)

    def _parse_markdown_with_metadata(self, text: str):
        if text.startswith("---"):
            parts = text.split("---", 2)

            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1]) or {}
                except:
                    metadata = {}

                content = parts[2].strip()

                classic = metadata.get("classic_rag", {})
                topics = classic.get("topics", [])

                metadata["topics"] = topics

                return metadata, content

        return {}, text

    def load_documents(self):
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[2]
        rag_db_path = project_root / "rag_db"

        docs = []

        for file_path in rag_db_path.rglob("*.md"):
            try:
                raw = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            meta, content = self._parse_markdown_with_metadata(raw)

            meta.update({
                "source": str(file_path),
                "file": file_path.name,
                "root_source": rag_db_path.name
            })

            docs.append(Document(
                page_content=content,
                metadata=meta
            ))

        print(f"Loaded {len(docs)} docs")
        return docs

    def chunk_documents(self, docs, chunk_size: int = 800, overlap_sentences: int = 2):

        chunks = []

        for d in docs:
            text = re.sub(r"\n{3,}", "\n\n", d.page_content).strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)

            current = []
            current_len = 0

            for sent in sentences:
                sent_len = len(sent)

                if current_len + sent_len > chunk_size:

                    chunk_text = " ".join(current).strip()

                    if len(chunk_text) > 120:
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=d.metadata
                        ))

                    current = current[-overlap_sentences:]
                    current_len = sum(len(x) for x in current)

                current.append(sent)
                current_len += sent_len

            if current:
                chunk_text = " ".join(current).strip()

                if len(chunk_text) > 120:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=d.metadata
                    ))

        print(f"Total chunks: {len(chunks)}")
        return chunks

    def retrieve(self, query: str) -> List[SearchResult]:
        return self.retriever.retrieve(query, top_k=10)

    def rerank(self, query: str, hits: List[SearchResult]) -> List[SearchResult]:
        return self.reranker.rerank(query, hits, top_n=5)

    def build_context(self, hits: List[SearchResult]) -> str:
        parts = []

        for h in hits:
            src = h.payload.get("file", "unknown")

            if len(h.text) < 100:
                continue

            parts.append(f"[SOURCE: {src}]\n{h.text}")

        return "\n\n---\n\n".join(parts)[:4500]

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


if __name__ == "__main__":

    rag = ClassicRAG()

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
