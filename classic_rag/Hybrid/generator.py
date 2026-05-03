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
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

class QwenClient(BaseLLMClient):

    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            top_p=0.8,
            temperature=0.0,
            repeat_penalty=1.15
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=200,
            stop=["\n\n\n", "Контекст:", "Вопрос:"]
        )

        answer = output["choices"][0]["text"].strip()

        answer = re.sub(r"<.*?>", "", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        return answer.strip()

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

        ЗАДАЧА:
        Ответь строго по контексту.

        ПРАВИЛА:
        - Используй ТОЛЬКО контекст
        - Не добавляй внешние знания
        - Если ответа нет → "Нет данных в контексте"
        - Не повторяй вопрос
        - Не рассуждай

        ФОРМАТ ОТВЕТА:
        Ответ:
        - пункт 1
        - пункт 2

        Источник:
        - Статья X ТК РФ

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

    def generate(self, query: str, context: str) -> str:
        context = self.cleaner.clean_context(context)

        if not context:
            return "Ответ:\nНет данных в контексте\n\nИсточник:\n-"

        prompt = self.prompt_builder.build(query, context)
        raw = self.llm.generate(prompt)

        return self._postprocess(raw)

    def _postprocess(self, text: str) -> str:

        text = re.sub(r"КОНЕЦ_ОТВЕТА.*", "", text, flags=re.DOTALL)

        if text.count("Ответ:") > 1:
            text = "Ответ:" + text.split("Ответ:")[1]

        text = re.sub(r"Вот ответ:?", "", text, flags=re.IGNORECASE)

        text = re.split(r"(Источник:)", text, maxsplit=1)

        if len(text) >= 3:
            text = text[0] + text[1] + text[2]

        text = re.sub(r"\n{3,}", "\n\n", text)

        if "Источник:" not in text:
            text += "\n\nИсточник:\n-"

        return text.strip()
