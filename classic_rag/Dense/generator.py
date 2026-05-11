import re
from abc import ABC, abstractmethod
from typing import List

from llama_cpp import Llama
from classic_rag.Dense.search_result import SearchResult


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
    def generate(
            self,
            query: str,
            context: str,
            hits: List[SearchResult]
    ) -> str:
        raise NotImplementedError



class ContextCleaner(BaseContextCleaner):
    def clean_context(self, text: str) -> str:
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text


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
                        "Ты юридический ассистент по Трудовому кодексу РФ. "
                        "Отвечай строго одним абзацем. "
                        "Запрещено: списки, рассуждения, reasoning, объяснения, вступления, служебные фразы. "
                        "Возвращай только готовый юридический ответ с источником статьи ТК РФ."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=0.8,
            repeat_penalty=1.1,
            max_tokens=250,
            stop=[
                "A:", "a:", "Answer:", "answer:", "Ответ:", "ответ:",
                "Okay", "Let's", "Reasoning", "Explanation", "Обоснование", "Анализ"
            ]
        )
        return output["choices"][0]["message"]["content"].strip()

    def close(self):
        self.llm = None



class LaborPromptBuilder(BasePromptBuilder):
    def build(self, query: str, context: str) -> str:
        return f"""
        Ты юридическая система по Трудовому кодексу РФ.
        
        Верни только готовый юридический ответ в одном абзаце.
        Строго запрещено:
        - рассуждения
        - reasoning
        - анализ
        - пояснения
        - списки
        - вступления
        - служебные фразы
        
        Используй контекст ниже, но только для точного ответа:
        {context}
        
        ВОПРОС:
        {query}
        
        ФИНАЛЬНЫЙ ОТВЕТ:
        """.strip()


class Generator(BaseGenerator):
    def __init__(self, llm: BaseLLMClient, prompt_builder: BasePromptBuilder, cleaner: BaseContextCleaner):
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
            if len(text) < 20:
                continue
            article = h.payload.get("article_number", "?")
            header = h.payload.get("header", "")
            parts.append(f"Статья {article} — {header}\n{text[:700]}")
        return "\n\n".join(parts)

    def _postprocess(self, text: str) -> str:
        if not text:
            return "Недостаточно данных."
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"(?i)^(a|answer|ответ):\s*", "", text)
        text = re.sub(r"(?i)(нужно ответить|сначала|важно|убеждаюсь|let'?s|okay|i need).*", "", text)
        text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) < 4:
            return "Недостаточно данных."
        return text

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass