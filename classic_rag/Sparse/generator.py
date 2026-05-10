import re
from abc import ABC, abstractmethod
from typing import List

from llama_cpp import Llama

from classic_rag.Sparse.search_result import SearchResult


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

        if not text:
            return ""

        text = re.sub(r"#+", "", text)
        text = re.sub(r"\*+", "", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"[ ]{2,}", " ", text)

        text = re.sub(r"---+", "", text)

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
                        "Ты юридический ассистент по Трудовому кодексу РФ.\n"
                        "Отвечай только финальным ответом.\n"
                        "Нельзя писать:\n"
                        "- reasoning\n"
                        "- анализ\n"
                        "- chain of thought\n"
                        "- внутренний монолог\n"
                        "- служебные фразы\n"
                        "- 'Хорошо'\n"
                        "- 'Проверю'\n"
                        "- 'Сначала'\n"
                        "- 'Пользователь спрашивает'\n"
                        "Ответ должен быть кратким юридическим абзацем."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.0,
            top_p=0.8,
            repeat_penalty=1.15,
            max_tokens=220,

            stop=[
                "<think>",
                "</think>",
                "Reasoning:",
                "Analysis:",
                "Проверю",
                "Хорошо",
                "Сначала",
                "A:",
                "Answer:",
            ]
        )

        return output["choices"][0]["message"]["content"].strip()

    def close(self):

        self.llm = None


class LaborPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:

        return f"""
        Ты юридическая система по Трудовому кодексу РФ.
        
        =====================
        ПРАВИЛА
        =====================
        
        Верни только готовый юридический ответ.
        
        Запрещено:
        - reasoning
        - chain of thought
        - анализ
        - объяснения
        - комментарии
        - внутренний монолог
        - служебные фразы
        
        Не начинай ответ со слов:
        - Хорошо
        - Проверю
        - Сначала
        - Пользователь спрашивает
        - Ответ:
        
        Формат:
        - один юридический абзац
        - без markdown
        - без списка
        - без пояснений
        - обязательно укажи статью ТК РФ
        
        =====================
        КОНТЕКСТ
        =====================
        
        {context}
        
        =====================
        ВОПРОС
        =====================
        
        {query}
        
        =====================
        ФИНАЛЬНЫЙ ОТВЕТ
        =====================
        """.strip()


class Generator(BaseGenerator):

    def __init__(
        self,
        llm,
        prompt_builder,
        cleaner
    ):

        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cleaner = cleaner

    def generate(
        self,
        query: str,
        context: str,
        hits: List[SearchResult]
    ) -> str:

        context = self.cleaner.clean_context(
            context or ""
        )

        if len(context) < 80:
            context = self._build_fallback_context(
                hits
            )

        if len(context) < 30:
            return "Недостаточно данных."

        prompt = self.prompt_builder.build(
            query,
            context
        )

        try:
            raw = self.llm.generate(prompt)

        except Exception as e:
            print("[LLM ERROR]", repr(e))
            return "Ошибка генерации ответа."

        answer = self._postprocess(raw)

        if (
            answer == "Недостаточно данных."
            and hits
        ):
            answer = self._fallback_answer(hits)

        return answer

    def _build_fallback_context(
        self,
        hits: List[SearchResult]
    ) -> str:

        parts = []

        for h in hits[:5]:

            text = (h.text or "").strip()

            if len(text) < 20:
                continue

            payload = h.payload or {}

            article = payload.get(
                "article_number",
                "?"
            )

            header = payload.get(
                "header",
                ""
            )

            parts.append(
                f"Статья {article}. {header}\n{text[:1000]}"
            )

        return "\n\n".join(parts)

    def _fallback_answer(
        self,
        hits: List[SearchResult]
    ) -> str:

        best = hits[0]

        payload = best.payload or {}

        article = payload.get(
            "article_number",
            "?"
        )

        text = (best.text or "").strip()

        text = re.sub(r"\s+", " ", text)

        return (
            f"{text} "
            f"Источник: Трудовой кодекс РФ, статья {article}."
        )

    def _postprocess(self, text: str) -> str:

        if not text:
            return "Недостаточно данных."

        text = re.sub(
            r"<.*?>",
            "",
            text,
            flags=re.DOTALL
        )

        text = re.sub(
            r"(?is)<think>.*?</think>",
            "",
            text
        )

        bad_patterns = [

            r"(?i)^a:\s*",
            r"(?i)^answer:\s*",

            r"(?i)^хорошо.*",
            r"(?i)^проверю.*",
            r"(?i)^сначала.*",
            r"(?i)^важно.*",

            r"(?i)^reasoning.*",
            r"(?i)^analysis.*",

            r"(?i)^пользователь спрашивает.*",

            r"(?i)^ответ должен.*",
        ]

        for pattern in bad_patterns:

            text = re.sub(
                pattern,
                "",
                text,
                flags=re.MULTILINE
            )

        text = re.sub(
            r"\n\s*[-•*]\s*",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if len(text.split()) < 5:
            return "Недостаточно данных."

        if "Источник:" not in text:
            text += " Источник: Трудовой кодекс РФ."

        return text.strip()

    def close(self):

        try:
            self.llm.close()

        except Exception:
            pass