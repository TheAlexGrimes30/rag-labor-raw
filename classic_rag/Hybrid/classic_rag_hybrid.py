import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        pass


class Reranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print("Loading Cross-Encoder Reranker...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return ranked_docs[:top_k]


class HybridRetriever(BaseRetriever):

    def __init__(self, documents: List[Document], alpha: float = 0.6, reranker: Reranker = None):
        print("Initializing HybridRetriever with optional Reranker...")
        self.documents = documents
        self.alpha = alpha
        self.reranker = reranker

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )
        self.db = FAISS.from_documents(documents, self.embeddings)

        self.corpus = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]
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
        sparse_scores = {self.documents[i].page_content: sparse_scores_array[i] for i in range(len(self.documents))}

        dense_norm = self.normalize(dense_scores)
        sparse_norm = self.normalize(sparse_scores)
        combined_scores = {}
        for doc in self.documents:
            content = doc.page_content
            d = dense_norm.get(content, 0)
            s = sparse_norm.get(content, 0)
            combined_scores[content] = self.alpha * d + (1 - self.alpha) * s

        ranked_docs = sorted(self.documents, key=lambda d: combined_scores.get(d.page_content, 0), reverse=True)

        unique = {}
        for d in ranked_docs:
            key = d.metadata["source"]
            if key not in unique:
                unique[key] = d
        docs_top = list(unique.values())[:k*2]

        if self.reranker:
            docs_top = self.reranker.rerank(query, docs_top, top_k=k)

        return docs_top


class QueryClassifier:

    def classify(self, query: str) -> str:
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

    def __init__(self):
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
        context = re.sub(r"#+\s*", "", context)
        context = re.sub(r"Вопрос:.*", "", context, flags=re.IGNORECASE)
        return context.strip()

    def build_prompt(self, query: str, context: str, query_type: str) -> str:
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
        return "\n".join(unique_lines).strip()

    def generate(self, query: str, context: str, query_type: str) -> str:
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
        if not self.documents:
            raise ValueError("No documents loaded!")

        print("Chunking documents...")
        self.chunks = self.chunk_documents(self.documents)

        print("Loading Reranker...")
        self.reranker = Reranker()

        print("Loading Hybrid retriever...")
        self.retriever = HybridRetriever(self.chunks, alpha=0.6, reranker=self.reranker)

        print("Loading generator...")
        self.generator = Generator()

        print("Loading classifier...")
        self.classifier = QueryClassifier()

    def load_documents(self):
        base_dir = Path(__file__).resolve().parent

        data_path = base_dir.parent.parent / "rag_db"
        dense_path = base_dir / "Hybrid"

        docs = []

        print(f"Scanning for documents in: {data_path}")
        print(f"Dense storage path: {dense_path}")

        if not data_path.exists():
            print(f"Directory not found: {data_path}")
            return docs

        if not dense_path.exists():
            print(f"WARNING: Hybrid folder not found: {dense_path}")

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
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        print(f"Total chunks: {len(chunks)}")
        return chunks

    def process_query(self, query):
        query = query.lower()
        query = re.sub(r"\bст\.\b", "статья", query)
        query = re.sub(r"\bтк\s*рф\b", "трудовой кодекс рф", query)
        return query

    def retrieve(self, query, query_type):
        k = 5 if query_type == "recommendation" else 3
        return self.retriever.retrieve(query, k=k)

    def build_context(self, docs):
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        return context[:4000]

    def ask(self, query):
        query_type = self.classifier.classify(query)
        query_processed = self.process_query(query)
        docs = self.retrieve(query_processed, query_type)
        context = self.build_context(docs)
        print("\n[DEBUG]")
        print("Query:", query)
        print("Type:", query_type)
        print("Retrieved:", [d.metadata["file_name"] for d in docs])
        answer = self.generator.generate(query, context, query_type)
        return RAGResponse(answer=answer, sources=[d.metadata for d in docs])


TEST_QUERIES = [
    "Можно ли употреблять алкоголь на рабочем месте?",
    "В 15 лет отказали в работе что делать?",
    "Объясни что такое испытательный срок"
]


def run_tests(rag: ClassicRAG):
    for q in TEST_QUERIES:
        print("\n========================")
        print("Q:", q)
        result = rag.ask(q)
        print("A:", result.answer)


if __name__ == "__main__":
    rag = ClassicRAG()
    run_tests(rag)
