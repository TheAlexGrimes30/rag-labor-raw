import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import yaml
from llama_cpp import Llama
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        pass


def parse_markdown_with_metadata(text: str):
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


class Reranker:
    def __init__(self):
        print("Loading Cross-Encoder Reranker...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: List[Document], top_k: int = 6):
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]



class HybridRetriever(BaseRetriever):

    def __init__(self, documents: List[Document], alpha: float = 0.7, reranker=None):
        print("Initializing HybridRetriever...")

        self.documents = documents
        self.alpha = alpha
        self.reranker = reranker

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        self.db = FAISS.from_documents(documents, self.embeddings)

        self.corpus = [self._tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def _normalize(self, scores: dict):
        vals = list(scores.values())
        if not vals:
            return scores

        mn, mx = min(vals), max(vals)
        if abs(mx - mn) < 1e-8:
            return {k: 0 for k in scores}

        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

    def retrieve(self, query: str, k: int = 3):

        dense = self.db.similarity_search_with_score(
            "query: " + query,
            k=min(40, len(self.documents))
        )

        dense_scores = {d.page_content: s for d, s in dense}

        tokenized = self._tokenize(query)
        bm25_scores_arr = self.bm25.get_scores(tokenized)

        bm25_scores = {
            self.documents[i].page_content: bm25_scores_arr[i]
            for i in range(len(self.documents))
        }
        dense_n = self._normalize(dense_scores)
        bm25_n = self._normalize(bm25_scores)

        combined = {}
        for doc in self.documents:
            c = doc.page_content
            combined[c] = (
                self.alpha * dense_n.get(c, 0)
                + (1 - self.alpha) * bm25_n.get(c, 0)
            )

        ranked = sorted(
            self.documents,
            key=lambda d: combined.get(d.page_content, 0),
            reverse=True
        )
        seen = set()
        filtered = []

        for d in ranked:
            key = d.metadata.get("id") or d.metadata.get("source")
            if key not in seen:
                seen.add(key)
                filtered.append(d)

        top = filtered[:25]
        if self.reranker:
            top = self.reranker.rerank(query, top, top_k=max(k, 6))

        return top[:k]


class Generator:

    def __init__(self):
        print("Loading LLM...")

        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            temperature=0.25,
            top_p=0.9,
            repeat_penalty=1.15,
        )

    def clean_context(self, text: str):
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def build_prompt(self, query: str, context: str):

        return f"""
        Ты — юридический помощник по трудовому праву Российской Федерации.
        
        ЗАДАЧА:
        Ответь на вопрос пользователя строго на основе контекста.
        
        ПРАВИЛА:
        - Используй контекст в первую очередь
        - Если в контексте нет точного ответа — скажи:
          "В предоставленном контексте нет точного ответа"
        - НЕ выдумывай статьи, номера законов и нормы
        - Если контекст неполный — дай общий юридически корректный ответ БЕЗ ссылок на статьи
        - Ответ должен быть кратким (3–6 предложений)
        - Пиши на русском языке
        
        КОНТЕКСТ:
        {context}
        
        ВОПРОС:
        {query}
        
        ОТВЕТ:
        """.strip()

    def generate(self, query: str, context: str):

        context = self.clean_context(context)
        prompt = self.build_prompt(query, context)

        output = self.llm(
            prompt,
            max_tokens=250,
            stop=["ВОПРОС:", "КОНТЕКСТ:", "ОТВЕТ:"]
        )

        return output["choices"][0]["text"].strip()



class ClassicRAG:

    def __init__(self):
        print("Loading documents...")
        self.documents = self.load_documents()

        print("Chunking...")
        self.chunks = self.chunk_documents(self.documents)

        self.reranker = Reranker()
        self.retriever = HybridRetriever(self.chunks, reranker=self.reranker)
        self.generator = Generator()

    def load_documents(self):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "..", "rag_db")

        docs = []

        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".md"):
                    full = os.path.join(root, f)

                    with open(full, "r", encoding="utf-8") as file:
                        raw = file.read()

                    meta, content = parse_markdown_with_metadata(raw)

                    meta.update({"source": full, "file": f})

                    docs.append(Document(page_content=content, metadata=meta))

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
