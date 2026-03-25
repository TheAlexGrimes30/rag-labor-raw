import os
import re
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        pass


class SparseRetriever(BaseRetriever):

    def __init__(self, documents: List[Document]):
        print("Initializing SparseRetriever (BM25)...")

        self.documents = documents

        self.corpus = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.corpus)

    def tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:k]

        docs = [self.documents[i] for i in top_k_idx]

        unique = {}
        for d in docs:
            key = d.metadata["source"]
            if key not in unique:
                unique[key] = d

        return list(unique.values())[:k]


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

        print("Loading Sparse retriever...")
        self.retriever = SparseRetriever(self.chunks)

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
                                metadata={
                                    "source": full_path,
                                    "file_name": file
                                }
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

                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    )
                )

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
        return context[:1000]

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
        print("\n========================")
        print("Q:", q)

        result = rag.ask(q)

        print("A:", result.answer)


if __name__ == "__main__":
    rag = ClassicRAG()
    run_tests(rag)
