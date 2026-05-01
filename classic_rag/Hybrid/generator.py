import re
from abc import ABC, abstractmethod

from llama_cpp import Llama


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
    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError


class ContextCleaner(BaseContextCleaner):

    def clean_context(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"#+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

class QwenClient(BaseLLMClient):

    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            top_p=0.7,
            temperature=0.0,
            repeat_penalty=1.2
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=150,
            stop=[
                "\n\n\n",
                "КОНТЕКСТ:",
                "ВОПРОС:",
                "Правила:",
                "Ты извлекаешь",
                "Пример:",
            ]
        )

        text = output["choices"][0]["text"]

        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

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
        Ты извлекаешь информацию из юридического текста.
    
        Правила:
        - Используй только текст из КОНТЕКСТА
        - Ничего не добавляй
        - Не объясняй
        - Не перефразируй
        - Копируй формулировки максимально точно
    
        Если ответа нет:
        Ответ:
        Нет данных в контексте
    
        Источник:
        -
    
        Формат ответа строго:
    
        Ответ:
        - ...
    
        Источник:
        - ...
    
        КОНТЕКСТ:
        {context}
    
        ВОПРОС:
        {query}
    
        Ответ:
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

    def generate(self, query: str, context: str) -> str:
        context = self.cleaner.clean_context(context)

        if not context:
            return self._empty()

        prompt = self.prompt_builder.build(query, context)
        raw = self.llm.generate(prompt)

        return self._postprocess(raw, context)

    def _empty(self):
        return "Ответ:\nНет данных в контексте\n\nИсточник:\n-"

    def _postprocess(self, text: str, context: str) -> str:

        if not text:
            return self._empty()

        text = text.strip()

        text = re.sub(
            r"(Таким образом.*|В этом примере.*|Ты понимаешь.*)",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        text = re.sub(r"^(Ответ на вопрос:.*?\n)", "", text, flags=re.IGNORECASE)

        if "Источник:" in text:
            text = text.split("Источник:")[0] + "Источник:" + text.split("Источник:")[1]

        if "Ответ:" not in text:
            text = "Ответ:\n" + text

        if "Источник:" not in text:
            text += "\n\nИсточник:\n-"

        text = self._enforce_extractive(text, context)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _enforce_extractive(self, text: str, context: str) -> str:
        """
        Удаляет строки, которых нет в контексте (анти-галлюцинация)
        """
        lines = text.split("\n")
        context_lower = context.lower()

        filtered = []

        for line in lines:
            clean_line = line.strip("- ").strip()

            if line.startswith("Ответ") or line.startswith("Источник"):
                filtered.append(line)
                continue

            if clean_line and clean_line.lower() in context_lower:
                filtered.append(line)

        if len(filtered) <= 2:
            return self._empty()

        return "\n".join(filtered)
