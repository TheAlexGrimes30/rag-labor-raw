import os
import re
from dataclasses import dataclass
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]


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
            do_sample=False
        )

    def generate(self, query: str, context: str) -> str:

        if not context:
            return "Недостаточно информации"

        prompt = f"""
Ты юридический ассистент по трудовому праву РФ.

Отвечай строго на основе контекста.
Если ответа нет — скажи "Недостаточно информации".

Контекст:
{context[:1500]}

Вопрос:
{query}

Ответ:
"""

        try:
            result = self.pipe(prompt)[0]["generated_text"]

            answer = result.split("Ответ:")[-1].strip()

            return answer

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

        print("Loading embeddings...")
        self.embeddings = self.get_embeddings()

        print("Building FAISS index...")
        self.vectorstore = self.build_faiss(self.chunks)

        print("Loading generator...")
        self.generator = Generator()

    def load_documents(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.abspath(os.path.join(BASE_DIR, "..", "rag_db"))

        docs = []

        print(f"Loading from: {data_path}")

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".md"):

                    full_path = os.path.join(root, file)

                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    if not text.strip():
                        continue

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

                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    )
                )

        print(f"Total chunks: {len(chunks)}")
        return chunks

    def get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base"
        )

    def build_faiss(self, docs):
        return FAISS.from_documents(docs, self.embeddings)

    def process_query(self, query):

        query = query.lower()

        query = re.sub(r"\bст\.\b", "статья", query)
        query = re.sub(r"\bтк\s*рф\b", "трудовой кодекс рф", query)

        return query

    def retrieve(self, query, k=5):

        query = "query: " + query

        docs = self.vectorstore.similarity_search(query, k=k)

        unique = []
        seen = set()

        for d in docs:
            key = d.page_content[:100]

            if key not in seen:
                unique.append(d)
                seen.add(key)

        return unique

    def build_context(self, docs):

        context = "\n\n".join(d.page_content for d in docs[:3])

        if len(context) < 50:
            return ""

        return context

    def ask(self, query):

        query = self.process_query(query)

        docs = self.retrieve(query)

        context = self.build_context(docs)

        answer = self.generator.generate(query, context)

        return RAGResponse(
            answer=answer,
            sources=[d.metadata for d in docs]
        )


TEST_QUERIES = [
    "Могу ли я работать с 14 лет?",
    "Со скольки лет можно работать?",
    "Ст. 63 ТК РФ",
    "Сколько дней отпуска положено?",
    "Что такое МРОТ?"
]


def run_tests(rag: ClassicRAG):

    for q in TEST_QUERIES:
        print("\n========================")
        print("Q:", q)

        result = rag.ask(q)

        print("A:", result.answer)

        print("Sources:")
        for s in result.sources:
            print(f"- {s['file_name']}")


if __name__ == "__main__":

    rag = ClassicRAG()

    run_tests(rag)
