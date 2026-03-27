import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

@dataclass
class RAGResponse:
    """
    Data structure representing the final response of the RAG system.

    Attributes:
        answer (str): Generated answer from the LLM.
        sources (List[Dict]): Metadata of documents used for generation.

    RU:
        Хранит итоговый ответ и источники, на основе которых он был сформирован.
    """

    answer: str
    sources: List[Dict]


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.

    RU:
        Базовый класс для всех retriever'ов (dense, sparse, hybrid).
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve top-k relevant documents for a query.

        Args:
            query (str): User query.
            k (int): Number of documents to retrieve.

        Returns:
            List[Document]: List of relevant documents.
        """

        pass


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using FAISS + embeddings.

    RU:
        Использует эмбеддинги + FAISS для поиска похожих документов.
    """

    def __init__(self, documents: List[Document]):
        """
        Initialize retriever and load/build FAISS index.

        Args:
            documents (List[Document]): Preprocessed document chunks.
        """

        print("Initializing DenseRetriever...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

        if os.path.exists("../faiss_index"):
            print("Loading FAISS index...")
            self.db = FAISS.load_local(
                "../faiss_index",
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
        """
        Retrieve relevant documents using MMR (Max Marginal Relevance).

        Args:
            query (str): User query.
            k (int): Number of documents.

        Returns:
            List[Document]: Unique top-k documents.

        RU:
            Используется MMR для балансировки релевантности и разнообразия.
        """

        query = "query: " + query
        docs = self.db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=10
        )
        unique = {}
        for d in docs:
            key = d.metadata["source"]
            if key not in unique:
                unique[key] = d
        return list(unique.values())[:k]


class Reranker:
    """
    Reranks retrieved documents using a cross-encoder.

    RU:
        Улучшает качество выдачи, переоценивая документы с помощью cross-encoder.
    """

    def __init__(self):
        """Load cross-encoder model."""

        print("Loading Cross-Encoder...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents by relevance.

        Args:
            query (str): User query.
            docs (List[Document]): Retrieved documents.
            top_k (int): Number of top documents.

        Returns:
            List[Document]: Reranked documents.
        """

        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = [d for d, s in scored_docs][:top_k]
        return reranked_docs


class QueryClassifier:
    """
    Classifies query into predefined categories.

    RU:
        Определяет тип запроса (вопрос, рекомендация, объяснение закона).
    """

    def classify(self, query: str) -> str:
        """
        Classify query type.

        Args:
            query (str): User query.

        Returns:
            str: Query type ("qa", "recommendation", "law_info").
        """

        q = query.lower()
        if any(word in q for word in [
            "что делать", "как поступить", "как быть",
            "отказали", "не платят", "уволили",
            "проблема", "нарушили", "задерживают"
        ]):
            return "recommendation"
        if any(word in q for word in [
            "что такое", "что означает", "объясни",
            "что регулирует", "что делает"
        ]):
            return "law_info"
        return "qa"


class Generator:
    """
    LLM-based answer generator.

    RU:
        Генерирует ответ на основе контекста и типа запроса.
    """

    def __init__(self):
        """Load LLM model and tokenizer."""

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.2,
            return_full_text=False
        )

    def clean_context(self, context: str) -> str:
        """
        Clean context text from noise.

        Args:
            context (str): Raw context.

        Returns:
            str: Cleaned context.
        """

        context = re.sub(r"#+\s*", "", context)
        context = re.sub(r"Вопрос:.*", "", context, flags=re.IGNORECASE)
        return context.strip()

    def build_prompt(self, query: str, context: str, query_type: str) -> str:
        """
        Build prompt for LLM.

        Args:
            query (str): User query.
            context (str): Retrieved context.
            query_type (str): Query type.

        Returns:
            str: Prompt string.
        """

        context_clean = self.clean_context(context)
        base_rules = """
        Используй ТОЛЬКО информацию из контекста.
        Если ответа нет — напиши: Недостаточно информации.
        Не придумывай факты.
        Отвечай кратко.
        """

        if query_type == "qa":
            return f"""
            Ты юридический ассистент по трудовому праву РФ.
            
            {base_rules}
            
            Формат:
            Да/Нет. Статья. Краткое пояснение.
            
            Контекст:
            {context_clean}
            
            Вопрос: {query}
            
            Ответ:
            """

        elif query_type == "recommendation":
            return f"""
            Ты юридический ассистент.
            
            {base_rules}
            
            ЗАПРЕЩЕНО:
            - писать приветствия
            
            Формат:
            1. Действие
            2. Действие
            Статья: ...
            
            Контекст:
            {context_clean}
            
            Ситуация: {query}
            
            Ответ:
            1.
            """

        elif query_type == "law_info":
            return f"""
            Ты юридический ассистент.
            
            {base_rules}
            
            Формат:
            Статья: ...
            Описание: ...
            
            Контекст:
            {context_clean}
            
            Вопрос: {query}
            
            Ответ:
            """

        return ""

    def postprocess(self, text: str) -> str:
        """
        Clean LLM output.

        Args:
            text (str): Raw generated text.

        Returns:
            str: Cleaned answer.

        RU:
            Убираем мусор, повторы, приветствия.
        """

        text = text.strip()
        if any(x in text.lower() for x in [
            "уважаемый", "гость", "добро пожаловать"
        ]):
            return "Недостаточно информации"
        text = re.split(r"Контекст:|Вопрос:", text)[0]
        lines = text.split("\n")
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in unique_lines:
                unique_lines.append(line)
        text = "\n".join(unique_lines)
        text = re.sub(r"(\b\w+\b)( \1\b)+", r"\1", text)
        return text.strip()

    def generate(self, query: str, context: str, query_type: str) -> str:
        """
        Generate final answer.

        Args:
            query (str): User query.
            context (str): Context.
            query_type (str): Query type.

        Returns:
            str: Final answer.
        """

        if not context.strip():
            return "Недостаточно информации"
        prompt = self.build_prompt(query, context, query_type)
        try:
            result = self.pipe(prompt)[0]["generated_text"]
            return self.postprocess(result)
        except Exception as e:
            print("LLM error:", e)
            return "Ошибка генерации"


class ClassicRAG:
    def __init__(self):
        print("Loading documents...")
        self.documents = self.load_documents()
        print("Chunking documents...")
        self.chunks = self.chunk_documents(self.documents)

        print("Loading retriever...")
        self.retriever = DenseRetriever(self.chunks)
        print("Loading reranker...")
        self.reranker = Reranker()
        print("Loading generator...")
        self.generator = Generator()
        print("Loading classifier...")
        self.classifier = QueryClassifier()

    def load_documents(self):
        base_dir = Path(__file__).resolve().parent

        data_path = base_dir.parent.parent / "rag_db"
        dense_path = base_dir / "Dense"

        docs = []

        print(f"Scanning for documents in: {data_path}")
        print(f"Dense storage path: {dense_path}")

        if not data_path.exists():
            print(f"Directory not found: {data_path}")
            return docs

        if not dense_path.exists():
            print(f"WARNING: Dense folder not found: {dense_path}")

        for path in data_path.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")

                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "relative_path": str(path.relative_to(data_path))
                        }
                    ))

            except Exception as e:
                print(f"Error reading {path}: {e}")

        print(f"Loaded {len(docs)} docs total")
        return docs

    def chunk_documents(self, docs):
        chunk_size = 500
        overlap = 100
        chunks = []
        for doc in docs:
            text = doc.page_content
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) < 100:
                    continue
                chunks.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ))
        print(f"Chunks: {len(chunks)}")
        return chunks

    def process_query(self, query):
        query = query.lower()
        query = re.sub(r"\bст\.\b", "статья", query)
        query = re.sub(r"\bтк\s*рф\b", "трудовой кодекс рф", query)
        return query

    def retrieve(self, query, query_type):
        if query_type == "recommendation":
            k = 5
        elif query_type == "law_info":
            k = 3
        else:
            k = 4
        docs = self.retriever.retrieve(query, k=k)
        docs = self.reranker.rerank(query, docs, top_k=k)
        return docs

    def build_context(self, docs):
        texts = [d.page_content for d in docs]
        return "\n\n---\n\n".join(texts)[:4000]

    def ask(self, query):
        query_type = self.classifier.classify(query)
        processed = self.process_query(query)
        docs = self.retrieve(processed, query_type)
        context = self.build_context(docs)

        print("\n[DEBUG]")
        print("Query:", query)
        print("Type:", query_type)
        print("Docs:", [d.metadata["file_name"] for d in docs])

        answer = self.generator.generate(query, context, query_type)

        return RAGResponse(
            answer=answer,
            sources=[d.metadata for d in docs]
        )


TEST_QUERIES = [
    "Можно ли употреблять алкоголь на рабочем месте?",
    "В 15 лет отказали в работе что делать?",
    "Объясни что такое испытательный срок"
]

def run_tests(rag: ClassicRAG):
    for q in TEST_QUERIES:
        print("\n====================")
        print("Q:", q)
        result = rag.ask(q)
        print("A:", result.answer)


if __name__ == "__main__":
    rag = ClassicRAG()
    run_tests(rag)
