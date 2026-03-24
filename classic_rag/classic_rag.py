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
            max_new_tokens=400,
            do_sample=False
        )

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return "Недостаточно информации"

        context_clean = re.sub(r"#+\s*", "", context)
        context_clean = re.sub(r"Вопрос:.*", "", context_clean, flags=re.IGNORECASE).strip()

        prompt = f"""
    Ты юридический ассистент по трудовому праву РФ.

    Отвечай КОРОТКО и строго на основе контекста.
    Начни ответ с "Да" или "Нет", укажи конкретный нормативно-правовой акт (например, статья 63 ТК РФ).
    Не добавляй лишние пояснения, только минимум информации для ответа.

    Контекст:
    {context_clean[:1500]}

    Вопрос:
    {query}

    Ответ:
    """

        try:
            result = self.pipe(prompt)[0]["generated_text"]
            answer = result.split("Ответ:")[-1].strip()
            answer = re.sub(r"Вопрос:.*", "", answer, flags=re.IGNORECASE).strip()
            answer = re.sub(r"\n{2,}", "\n", answer)
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

        unique = {}
        for d in docs:
            key = d.metadata["source"]
            if key not in unique:
                unique[key] = d

        return list(unique.values())[:3]

    def build_context(self, docs):
        cleaned_chunks = []
        for d in docs:
            text = d.page_content.strip()
            if "Вопрос:" in text:
                text = text.split("Вопрос:")[0]
            cleaned_chunks.append(text)
        context = "\n\n---\n\n".join(cleaned_chunks)
        return context[:1500]

    def ask(self, query):
        original_query = query
        query = self.process_query(query)
        docs = self.retrieve(query)
        context = self.build_context(docs)

        print("\n[DEBUG]")
        print("Query:", original_query)
        print("Retrieved:", [d.metadata["file_name"] for d in docs])
        print("Context length:", len(context))

        answer = self.generator.generate(original_query, context)
        return RAGResponse(
            answer=answer,
            sources=[d.metadata for d in docs]
        )


TEST_QUERIES = [
    "Можно ли употреблять алкоголь на рабочем месте?"
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
