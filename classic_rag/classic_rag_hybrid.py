import heapq
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

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


class Reranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        print("[Reranker] Loading Cross-Encoder model...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = heapq.nlargest(top_k, zip(scores, docs), key=lambda x: x[0])
        return [d for _, d in ranked]


class HybridRetriever(BaseRetriever):

    def __init__(self, documents: List[Document], alpha: float = 0.5, reranker: Reranker = None):
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
        print("Loading LLM via llama.cpp...")

        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            chat_format=None
        )

    def clean_context(self, context: str) -> str:
        context = re.sub(r"#+\s*", "", context)
        context = re.sub(r"Вопрос:.*", "", context, flags=re.IGNORECASE)

        context = re.sub(r"(?i)instruction.*", "", context)
        context = re.sub(r"(?i)response.*", "", context)
        context = re.sub(r"<\|.*?\|>", "", context)

        context = re.sub(r"\n{2,}", "\n", context)

        return context.strip()

    def build_prompt(self, query: str, context: str, query_type: str) -> str:
        context = self.clean_context(context)

        base_rules = """
        Отвечай ТОЛЬКО по контексту.
        Если ответа нет — напиши: Недостаточно информации.
        Не придумывай факты.
        Отвечай кратко и по делу.
        """

        examples = """
        ### Примеры:
        
        Вопрос: Можно ли употреблять алкоголь на рабочем месте?
        Ответ: Нет, запрещено Трудовым кодексом РФ.
        
        Вопрос: В 15 лет отказали в работе что делать?
        Ответ:
        1. Обратиться в трудовую инспекцию.
        2. Обратиться в органы опеки.
        
        Вопрос: Что такое испытательный срок?
        Ответ: Испытательный срок — период проверки работника перед окончательным трудоустройством.
        """

        if query_type == "qa":
            task = "Дай точный юридический ответ (да/нет + краткое пояснение)."

        elif query_type == "recommendation":
            task = "Дай пошаговые рекомендации (1, 2, 3)."

        elif query_type == "law_info":
            task = "Дай краткое определение юридического термина."

        else:
            task = "Ответь на вопрос."

        prompt = f"""
        Ты юрист по трудовому праву РФ.
        
        {base_rules}
        
        {examples}
        
        ### Задание:
        {task}
        
        ---
        
        ### Контекст:
        {context}
        
        ---
        
        ### Вопрос:
        {query}
        
        ---
        
        ### Ответ:
        """
        return prompt

    def postprocess(self, text: str) -> str:
        text = text.strip()

        text = re.split(r"###|Instruction|Response|Контекст:", text)[0]

        lines = []
        for line in text.split("\n"):
            line = line.strip()

            if (
                line
                and len(line) > 3
                and not line.lower().startswith(("вопрос", "контекст"))
                and line not in lines
            ):
                lines.append(line)

        return "\n".join(lines).strip()

    def generate(self, query: str, context: str, query_type: str) -> str:
        if not context.strip():
            return "Недостаточно информации"

        prompt = self.build_prompt(query, context, query_type)

        try:
            result = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.2,
                stop=["###", "</s>"]
            )

            text = result["choices"][0]["text"]
            return self.postprocess(text)

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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(BASE_DIR, "..", "rag_db"))
        docs = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text.strip():
                        docs.append(
                            Document(
                                page_content=text.strip(),
                                metadata={"source": full_path, "file_name": file}
                            )
                        )
        print(f"Loaded {len(docs)} documents")
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
        return context

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
