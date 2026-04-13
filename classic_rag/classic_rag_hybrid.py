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
            yaml_part = parts[1]
            content_part = parts[2]

            try:
                metadata = yaml.safe_load(yaml_part) or {}
            except Exception:
                metadata = {}

            return metadata, content_part.strip()

    return {}, text

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print("Loading Cross-Encoder Reranker...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked_docs = [
            doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        ]
        return ranked_docs[:top_k]

class HybridRetriever(BaseRetriever):

    def __init__(self, documents: List[Document], alpha: float = 0.6, reranker: Reranker = None):
        print("Initializing HybridRetriever...")
        self.documents = documents
        self.alpha = alpha
        self.reranker = reranker

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        self.db = FAISS.from_documents(documents, self.embeddings)

        self.corpus = [self.tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def normalize(self, scores_dict):
        values = list(scores_dict.values())
        if not values:
            return scores_dict

        min_v, max_v = min(values), max(values)
        if max_v - min_v < 1e-8:
            return {k: 0 for k in scores_dict}

        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def retrieve(self, query: str, k: int = 3) -> List[Document]:

        dense_results = self.db.similarity_search_with_score(
            "query: " + query,
            k=len(self.documents)
        )
        dense_scores = {doc.page_content: score for doc, score in dense_results}

        tokenized_query = self.tokenize(query)
        sparse_scores_array = self.bm25.get_scores(tokenized_query)
        sparse_scores = {
            self.documents[i].page_content: sparse_scores_array[i]
            for i in range(len(self.documents))
        }

        dense_norm = self.normalize(dense_scores)
        sparse_norm = self.normalize(sparse_scores)

        combined_scores = {}
        for doc in self.documents:
            content = doc.page_content
            d = dense_norm.get(content, 0)
            s = sparse_norm.get(content, 0)
            combined_scores[content] = self.alpha * d + (1 - self.alpha) * s

        ranked_docs = sorted(
            self.documents,
            key=lambda d: combined_scores.get(d.page_content, 0),
            reverse=True
        )

        unique = {}
        for d in ranked_docs:
            key = d.metadata.get("id", d.metadata.get("source"))
            if key not in unique:
                unique[key] = d

        docs_top = list(unique.values())[:k * 2]

        if self.reranker:
            docs_top = self.reranker.rerank(query, docs_top, top_k=k)

        return docs_top


class QueryClassifier:
    def classify(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in ["что делать", "не платят", "уволили", "нарушили"]):
            return "recommendation"

        if any(w in q for w in ["что такое", "объясни", "что означает"]):
            return "law_info"

        return "qa"


class Generator:
    def __init__(self):
        print("Loading LLM...")

        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8
        )

    def clean_context(self, context: str):
        context = re.sub(r"#+\s*", "", context)
        return context.strip()

    def build_prompt(self, query, context):
        return f"""
        Ты юридический ассистент.
        
        Используй только контекст.
        
        Контекст:
        {context}
        
        Вопрос: {query}
        
        Ответ:
        """

    def generate(self, query, context):
        prompt = self.build_prompt(query, context)

        result = self.llm(prompt, max_tokens=200)
        return result["choices"][0]["text"].strip()


class ClassicRAG:

    def __init__(self):
        print("Loading documents...")
        self.documents = self.load_documents()

        print("Chunking...")
        self.chunks = self.chunk_documents(self.documents)

        self.reranker = Reranker()
        self.retriever = HybridRetriever(self.chunks, reranker=self.reranker)
        self.generator = Generator()
        self.classifier = QueryClassifier()

    def load_documents(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, "..", "rag_db")

        docs = []

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)

                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read()

                    metadata, content = parse_markdown_with_metadata(raw)

                    metadata.update({
                        "source": path,
                        "file_name": file
                    })

                    docs.append(
                        Document(
                            page_content=content,
                            metadata=metadata
                        )
                    )

        print(f"Loaded {len(docs)} documents")
        return docs

    def chunk_documents(self, docs):
        chunks = []

        for doc in docs:
            sections = re.split(r"\n### |\n## ", doc.page_content)

            for sec in sections:
                sec = sec.strip()
                if len(sec) < 100:
                    continue

                chunks.append(
                    Document(
                        page_content=sec,
                        metadata=doc.metadata
                    )
                )

        print(f"Total chunks: {len(chunks)}")
        return chunks

    def retrieve(self, query):
        return self.retriever.retrieve(query, k=3)

    def build_context(self, docs):
        parts = []

        for d in docs:
            article_id = d.metadata.get("id", "")
            parts.append(f"[{article_id}]\n{d.page_content}")

        return "\n\n---\n\n".join(parts)[:4000]

    def ask(self, query):
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

        "что считается дискриминацией в труде",
        "можно ли отказать в работе из-за возраста"
    ]

    for q in questions:
        print("\nQ:", q)
        res = rag.ask(q)
        print("A:", res.answer)