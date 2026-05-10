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

        return text.strip()


class QwenClient(BaseLLMClient):

    def __init__(self, model_path: str):

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            temperature=0.1,
            top_p=0.7,
            repeat_penalty=1.2
        )

    def generate(self, prompt: str) -> str:

        output = self.llm(
            prompt,
            max_tokens=180,
            stop=[
                "КОНТЕКСТ:",
                "Контекст:",
                "Запрос:",
                "ВОПРОС:",
                "\n[Источник",
                "\nСтатья:"
            ]
        )

        text = output["choices"][0]["text"]

        return text.strip()

    def close(self):

        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass

            self.llm = None


class LaborPromptBuilder(BasePromptBuilder):

    def build(self, query: str, context: str) -> str:

        return f"""
        Ты — юридический ассистент по трудовому праву РФ.
        
        Используй ТОЛЬКО информацию из контекста.
        
        ЗАПРЕЩЕНО:
        - придумывать информацию
        - объяснять свои действия
        - писать рассуждения
        - писать "Обоснование"
        - писать "Вывод"
        - повторять вопрос
        - повторять контекст
        - упоминать prompt
        - использовать фразы:
          "в контексте"
          "в предоставленных источниках"
          "недостаточно информации"
          "проверьте"
          "внимательно"
        - писать статьи, которых нет в контексте
        
        ПРАВИЛА ОТВЕТА:
        - ответ должен быть кратким
        - ответ должен быть юридически точным
        - если есть перечисление — используй список
        - если перечисления нет — используй обычный текст
        - не дублируй информацию
        - обязательно укажи норму права
        
        ЕСЛИ ОТВЕТА НЕТ:
        Недостаточно данных.
        
        КОНТЕКСТ:
        {context}
        
        ВОПРОС:
        {query}
        
        ОТВЕТ:
        """.strip()


class Generator(BaseGenerator):

    def __init__(
            self,
            llm: BaseLLMClient,
            prompt_builder: BasePromptBuilder,
            cleaner: BaseContextCleaner
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

        context = self.cleaner.clean_context(context)

        if not context:
            return "Недостаточно данных."

        prompt = self.prompt_builder.build(
            query=query,
            context=context
        )

        raw = self.llm.generate(prompt)

        cleaned = self._postprocess(raw)

        sources = self._build_sources(hits)

        return f"{cleaned}\n\nИсточник:\n{sources}"

    def _postprocess(self, text: str) -> str:

        text = re.sub(r"<.*?>", "", text)

        text = re.sub(r"\*+", "", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        garbage_patterns = [
            r"(?i)обоснование:.*",
            r"(?i)вывод:.*",
            r"(?i)в контексте.*",
            r"(?i)в предоставленных источниках.*",
            r"(?i)проверьте.*",
            r"(?i)внимательно.*",
            r"(?i)ты —.*",
            r"(?i)запрещено:.*",
            r"(?i)правила ответа:.*",
            r"(?i)контекст:.*",
            r"(?i)вопрос:.*",
            r"(?i)ответ:.*",
        ]

        for pattern in garbage_patterns:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.MULTILINE
            )

        text = self._remove_duplicate_lines(text)

        text = self._remove_duplicate_sentences(text)

        text = self._normalize_lists(text)

        text = text.strip()

        if not text:
            text = "Недостаточно данных."

        return text

    def _remove_duplicate_lines(
            self,
            text: str
    ) -> str:

        seen = set()

        result = []

        for line in text.splitlines():

            normalized = re.sub(
                r"\s+",
                " ",
                line.strip().lower()
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)

            result.append(line.strip())

        return "\n".join(result)

    def _remove_duplicate_sentences(
            self,
            text: str
    ) -> str:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        seen = set()

        result = []

        for sentence in sentences:

            normalized = re.sub(
                r"\s+",
                " ",
                sentence.lower()
            ).strip()

            if len(normalized) < 8:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)

            result.append(sentence.strip())

        return " ".join(result)

    def _normalize_lists(
            self,
            text: str
    ) -> str:

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith("-"):

                line = re.sub(
                    r"^-+\s*",
                    "- ",
                    line
                )

            lines.append(line)

        bullet_count = sum(
            1 for x in lines
            if x.startswith("- ")
        )

        if bullet_count <= 1:

            text = " ".join(
                line.replace("- ", "")
                for line in lines
            )

            return text.strip()

        return "\n".join(lines)

    def _build_sources(
            self,
            hits: List[SearchResult]
    ) -> str:

        seen = set()

        sources = []

        for h in hits:

            article = h.payload.get(
                "article_number"
            )

            if not article:
                continue

            source = (
                f"- Трудовой кодекс РФ, статья {article}"
            )

            if source in seen:
                continue

            seen.add(source)

            sources.append(source)

        if not sources:
            return "-"

        return "\n".join(sources)
