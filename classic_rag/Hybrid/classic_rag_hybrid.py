import re
from pathlib import Path
import yaml
from langchain_core.documents import Document
from classic_rag.Hybrid.generator import Generator
from classic_rag.Hybrid.rag_config import RAGResponse
from classic_rag.Hybrid.reranker import Reranker
from classic_rag.Hybrid.retriever import HybridRetriever


class ClassicRAG:

    def __init__(self):
        print("Loading documents...")
        self.documents = self.load_documents()

        print("Chunking...")
        self.chunks = self.chunk_documents(self.documents)

        self.reranker = Reranker()
        self.retriever = HybridRetriever(self.chunks, reranker=self.reranker)
        self.generator = Generator()

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

    def retrieve(self, query):
        return self.retriever.retrieve(query, k=3)

    def build_context(self, docs):
        parts = []

        for d in docs:
            src = d.metadata.get("file", "unknown")

            if len(d.page_content) < 100:
                continue

            parts.append(f"[SOURCE: {src}]\n{d.page_content}")

        return "\n\n---\n\n".join(parts)[:4500]

    def ask(self, query: str) -> RAGResponse:

        docs = self.retrieve(query)
        context = self.build_context(docs)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[d.metadata for d in docs]
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
