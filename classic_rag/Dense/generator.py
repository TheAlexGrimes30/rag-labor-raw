import re
from abc import ABC, abstractmethod
from typing import List, Set

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
    def generate(self, query: str, context: str, hits: List[SearchResult]) -> str:
        raise NotImplementedError


# =========================================================
# CONTEXT CLEANER
# =========================================================

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
                        "Ты юридический ассистент по Трудовому кодексу РФ.\n"
                        "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.\n"
                        "СТРОГО ЗАПРЕЩЕНО:\n"
                        "- английский язык\n"
                        "- рассуждения\n"
                        "- reasoning\n"
                        "- объяснения\n"
                        "- списки\n"
                        "- любые служебные фразы\n"
                        "- выход за пределы контекста\n"
                        "Верни только один юридический абзац."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.0,
            top_p=0.7,
            repeat_penalty=1.2,
            max_tokens=220,

            stop=[
                "Reasoning",
                "Explanation",
                "Analysis",
                "A:",
                "Answer:",
                "Let's",
                "Okay"
            ]
        )

        return output["choices"][0]["message"]["content"].strip()

    def close(self):
        self.llm = None



class LaborPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:

        return f"""
        Ты отвечаешь ТОЛЬКО на основе предоставленного контекста из Трудового кодекса РФ.
        
        ⚠️ ПРАВИЛА:
        - нельзя использовать внешние знания
        - нельзя додумывать статьи
        - нельзя рассуждать
        - нельзя объяснять
        - если ответа нет в контексте → скажи "Недостаточно данных"
        
        ФОРМАТ:
        - один абзац
        - только юридическая норма
        - без списков
        
        КОНТЕКСТ:
        {context}
        
        ВОПРОС:
        {query}
        
        ОТВЕТ:
        """.strip()

class Generator(BaseGenerator):

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

        return self._postprocess(raw, hits)

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


    def _extract_articles(self, hits: List[SearchResult]) -> Set[str]:
        return {
            str(h.payload.get("article_number"))
            for h in hits
            if h.payload.get("article_number")
        }

    def _postprocess(self, text: str, hits: List[SearchResult]) -> str:

        if not text:
            return "Недостаточно данных."

        text = text.strip()

        # remove artifacts
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"(?i)(reasoning|analysis|explanation|let's|okay|i need).*", "", text)

        # remove bullets
        text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)

        text = re.sub(r"\s+", " ", text).strip()

        # ❗ HARD: no English allowed
        if re.search(r"[A-Za-z]", text):
            return "Недостаточно данных."

        # ❗ HARD: article sanity check
        allowed_articles = self._extract_articles(hits)

        found_articles = re.findall(r"\b\d+\b", text)

        if found_articles:
            if not any(a in allowed_articles for a in found_articles):
                return "Недостаточно данных."

        if len(text.split()) < 4:
            return "Недостаточно данных."

        return text

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass