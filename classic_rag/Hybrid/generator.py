import re
from abc import ABC, abstractmethod
from typing import List

from llama_cpp import Llama
from classic_rag.Hybrid.search_result import SearchResult



class BaseLLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(self, query: str, context: str) -> str:
        raise NotImplementedError


class BaseContextCleaner(ABC):

    @abstractmethod
    def clean_context(self, text: str) -> str:
        raise NotImplementedError


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, query: str, context: str, hits: List[SearchResult]) -> str:
        raise NotImplementedError


class ContextCleaner(BaseContextCleaner):

    def clean_context(self, text: str) -> str:
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


class QwenClient(BaseLLMClient):

    def __init__(self, model_path: str):

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )

    def generate(self, prompt: str) -> str:

        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты юридический ассистент по ТК РФ. "
                        "Отвечай строго по контексту. "
                        "НЕ ПИШИ рассуждения, шаги или объяснения. "
                        "Возвращай только финальный ответ."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=320,
            temperature=0.1,
            top_p=0.85,
            repeat_penalty=1.15,
            stop=[
                "\nОбоснование",
                "\nАнализ",
                "\nПояснение",
                "\nReasoning",
                "\nExplanation"
            ]
        )

        return output["choices"][0]["message"]["content"].strip()


class LaborPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:

        return f"""
        Ты юридическая система по Трудовому кодексу РФ.
        
        =====================
        ПРАВИЛА
        =====================
        
        - Отвечай строго по контексту
        - НЕ добавляй рассуждения
        - НЕ объясняй процесс
        - НЕ используй слова: "анализ", "обоснование", "пояснение"
        - Дай ТОЛЬКО финальный ответ
        - Если есть список — используй список
        
        ВАЖНО:
        НЕ ПИШИ НИЧЕГО КРОМЕ ОТВЕТА
        
        =====================
        КОНТЕКСТ
        =====================
        
        {context}
        
        =====================
        ВОПРОС
        =====================
        
        {query}
        
        =====================
        ОТВЕТ
        =====================
        """.strip()


class Generator:

    def __init__(self, llm, prompt_builder, cleaner):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cleaner = cleaner


    def generate(self, query: str, context: str, hits: List[SearchResult]) -> str:

        context = self.cleaner.clean_context(context or "")

        if len(context) < 80:
            context = self._build_fallback_context(hits)

        if len(context) < 30:
            return "Недостаточно данных."

        prompt = self.prompt_builder.build(query, context)

        try:
            raw = self.llm.generate(prompt)
        except Exception:
            return "Ошибка генерации ответа."

        return self._postprocess(raw)


    def _build_fallback_context(self, hits: List[SearchResult]) -> str:

        parts = []

        for h in hits[:5]:

            text = (h.text or "").strip()
            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")

            if len(text) < 20:
                continue

            parts.append(
                f"Статья {article} — {header}\n{text[:700]}"
            )

        return "\n\n".join(parts)


    def _postprocess(self, text: str) -> str:

        if not text:
            return "Недостаточно данных."

        text = re.sub(r"<.*?>", "", text).strip()

        cut_markers = [
            "обоснование",
            "анализ",
            "пояснение",
            "комментарий",
            "reasoning",
            "explanation"
        ]

        lower = text.lower()
        cut_pos = len(text)

        for m in cut_markers:
            idx = lower.find(m)
            if idx != -1:
                cut_pos = min(cut_pos, idx)

        text = text[:cut_pos].strip()

        lines = []
        bullet_count = 0

        for line in text.splitlines():
            l = line.strip()
            if not l:
                continue

            if re.match(r"^[-•*]\s+", l):
                l = re.sub(r"^[-•*]\s+", "- ", l)
                bullet_count += 1

            lines.append(l)

        text = "\n".join(lines).strip()

        if len(text.split()) < 4:
            return "Недостаточно данных."

        return text

    def close(self):
        print("Shutting down RAG...")

        try:
            if hasattr(self.llm, "llm"):
                self.llm = None

        except Exception as e:
            print("[WARN] LLM cleanup error:", repr(e))

        try:
            if hasattr(self, "client"):
                self.client.close()
        except Exception as e:
            print("[WARN] Qdrant close error:", repr(e))