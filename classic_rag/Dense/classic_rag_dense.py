import os
import re
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from sentence_transformers import CrossEncoder

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        pass


class DenseRetriever(BaseRetriever):
    def __init__(self, documents: List[Document]):
        print("Initializing DenseRetriever...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        if os.path.exists("faiss_index"):
            print("Loading FAISS index...")
            self.db = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Building FAISS index...")
            for doc in documents:
                doc.page_content = "passage: " + doc.page_content

            self.db = FAISS.from_documents(documents, self.embeddings)
            self.db.save_local("faiss_index")

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        query = "query: " + query

        docs = self.db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=10
        )

        unique = {}
        for d in docs:
            key = d.metadata.get("article_id", d.metadata["source"])
            if key not in unique:
                unique[key] = d

        return list(unique.values())[:k]

class Reranker:
    def __init__(self):
        print("Loading Cross-Encoder...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored_docs][:top_k]


class QueryClassifier:
    def classify(self, query: str) -> str:
        q = query.lower()

        if any(w in q for w in ["что делать", "как поступить", "не платят", "уволили"]):
            return "recommendation"

        if any(w in q for w in ["что такое", "объясни", "что регулирует"]):
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
            n_threads=8,
            n_gpu_layers=0
        )

    def clean_context(self, context: str) -> str:
        context = re.sub(r"---.*?---", "", context, flags=re.DOTALL)

        context = re.sub(r"#", "", context)

        context = re.sub(r"\n{2,}", "\n", context)

        return context.strip()

    def extract_article(self, context: str) -> str:
        match = re.search(r"Статья\s+(\d+)", context)
        return match.group(1) if match else "?"

    def build_prompt(self, query: str, context: str, query_type: str) -> str:
        context = self.clean_context(context)

        base_rules = """
        Ты юридический ассистент по ТК РФ.
        
        Правила:
        1. Используй ТОЛЬКО контекст
        2. НЕ придумывай
        3. Отвечай кратко
        4. Укажи статью
        """

        if query_type == "qa":
            return f"""
            {base_rules}
            
            Формат:
            Да/Нет
            Статья: ...
            Ответ: ...
            
            Контекст:
            {context}
            
            Вопрос: {query}
            Ответ:
            """

        elif query_type == "law_info":
            return f"""
            {base_rules}
            
            Формат:
            Статья: ...
            Описание: ...
            
            Контекст:
            {context}
            
            Вопрос: {query}
            Ответ:
            """

        elif query_type == "recommendation":
            return f"""
            {base_rules}
            
            Формат:
            1. ...
            2. ...
            Статья: ...
            
            Контекст:
            {context}
            
            Ситуация: {query}
            Ответ:
            """

    def generate(self, query, context, query_type):
        if not context.strip():
            return "Недостаточно информации"

        prompt = self.build_prompt(query, context, query_type)

        result = self.llm(
            prompt,
            max_tokens=200,
            temperature=0.1,
            stop=["Контекст:", "Вопрос:"]
        )

        return result["choices"][0]["text"].strip()


class ClassicRAG:

    def __init__(self):
        print("Loading documents...")
        self.documents = self.load_documents()

        print("Chunking...")
        self.chunks = self.chunk_documents(self.documents)

        print("Retriever...")
        self.retriever = DenseRetriever(self.chunks)

        print("Reranker...")
        self.reranker = Reranker()

        print("Generator...")
        self.generator = Generator()

        print("Classifier...")
        self.classifier = QueryClassifier()

    def split_md(self, text: str):
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if match:
            metadata = yaml.safe_load(match.group(1))
            content = match.group(2)
        else:
            metadata = {}
            content = text
        return metadata, content

    def load_documents(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, "..", "rag_db")

        docs = []

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)

                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()

                    metadata, content = self.split_md(text)

                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": path,
                            "file_name": file,
                            "article_id": metadata.get("id")
                        }
                    ))

        print(f"Loaded: {len(docs)}")
        return docs

    def chunk_documents(self, docs):
        chunks = []

        for doc in docs:
            text = doc.page_content

            parts = re.split(r"\n---\n|###|##", text)

            for part in parts:
                if len(part.strip()) < 100:
                    continue

                chunks.append(Document(
                    page_content=part.strip(),
                    metadata=doc.metadata
                ))

        print(f"Chunks: {len(chunks)}")
        return chunks

    def process_query(self, query):
        query = query.lower()
        query = re.sub(r"\bст\.\b", "статья", query)
        return query

    def retrieve(self, query, query_type):
        k = 5 if query_type == "recommendation" else 3

        docs = self.retriever.retrieve(query, k=k)
        docs = self.reranker.rerank(query, docs, top_k=k)

        return docs

    def build_context(self, docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def ask(self, query):
        q_type = self.classifier.classify(query)
        processed = self.process_query(query)

        docs = self.retrieve(processed, q_type)
        context = self.build_context(docs)

        print("\n[DEBUG]")
        print("Query:", query)
        print("Docs:", [d.metadata["file_name"] for d in docs])

        answer = self.generator.generate(query, context, q_type)

        return RAGResponse(
            answer=answer,
            sources=[d.metadata for d in docs]
        )

TEST_QUERIES = [
    "какие цели трудового законодательства",
    "что такое свобода труда",
    "что считается дискриминацией"
]


def run_tests(rag):
    for q in TEST_QUERIES:
        print("\n====================")
        print("Q:", q)
        result = rag.ask(q)
        print("A:", result.answer)


if __name__ == "__main__":
    rag = ClassicRAG()
    run_tests(rag)
