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
                        "Ты модуль извлечения юридических норм.\n"
                        "НЕ веди диалог.\n"
                        "НЕ объясняй процесс.\n"
                        "НЕ используй слова типа: 'сначала', 'проверяю', 'анализирую'.\n"
                        "Выводи только готовый юридический ответ.\n"
                        "Никаких рассуждений и пояснений."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.15,
            top_p=0.85,
            repeat_penalty=1.1,
            max_tokens=512
        )

        return output["choices"][0]["message"]["content"].strip()


class LaborPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:
        return f"""
        Ты извлекаешь юридический ответ ТОЛЬКО из контекста.

        ПРАВИЛА:
        - НЕ объясняй ход мыслей
        - НЕ используй слова: "сначала", "проверяю", "анализ"
        - НЕ добавляй внешние знания
        - НЕ рассуждай

        ФОРМАТ:
        - 3–6 предложений
        - юридически точный текст
        - без списков
        - без вступлений

        ЕСЛИ НЕТ ДАННЫХ:
        Ответ: "Нет данных в предоставленных источниках"

        =====================
        КОНТЕКСТ
        =====================
        {context}

        =====================
        ВОПРОС
        =====================
        {query}

        =====================
        ОТВЕТ:
        =====================
        """.strip()


class Generator(BaseGenerator):

    def __init__(self, llm, prompt_builder, cleaner):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cleaner = cleaner

    def generate(
            self,
            query: str,
            context: str,
            hits: List[SearchResult]
    ) -> str:

        context = self.cleaner.clean_context(context or "")

        if len(context) < 80:
            context = self._build_fallback_context(hits)

        if len(context) < 30:
            return "Недостаточно данных."

        prompt = self.prompt_builder.build(query, context)

        try:
            raw = self.llm.generate(prompt)
        except Exception as e:
            print(f"[GENERATION ERROR] {e}")
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

        text = re.sub(r"<.*?>", "", text).strip()

        text = re.sub(r"(?i)^(a|answer|ответ):\s*", "", text)

        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()

        if len(text.split()) < 6:
            return "Недостаточно данных."

        if re.search(r"(?i)\b(сначала|проверяю|анализирую|рассмотрю)\b", text):
            text = re.sub(r"(?i)\b(сначала|проверяю|анализирую|рассмотрю).*", "", text).strip()

        return text

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass