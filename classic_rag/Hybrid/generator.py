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
            top_p=0.9,
            repeat_penalty=1.2,
            temperature=0.1
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=180,
            stop=["\n\n\n", "ВОПРОС:", "КОНТЕКСТ:"]
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
        Ты — эксперт по трудовому праву РФ.

        Задача: ответить на вопрос на основе контекста.

        ФОРМАТ:
        Ответ:
        1. Да / Нет / Частично / Факт
        2. Краткое обоснование
        3. Статья ТК РФ или "-"

        ПРАВИЛА:
        - Используй контекст как основной источник
        - Допускается логический вывод, если нет прямого определения
        - Не добавляй лишнего текста

        ПРИМЕР:

        КОНТЕКСТ:
        Свобода труда означает право свободно распоряжаться своими способностями к труду,
        выбирать род деятельности и профессию.

        ВОПРОС:
        что такое свобода труда

        Ответ:
        1. Факт
        2. Свобода труда — это право свободно распоряжаться своими способностями к труду
        3. Статья 2 ТК РФ

        ЕСЛИ НЕТ ДАННЫХ:
        Ответ:
        Факт
        В предоставленном контексте отсутствует информация
        -

        КОНТЕКСТ:
        {context}

        ВОПРОС:
        {query}

        КОНЕЦ_ОТВЕТА
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
            return "Факт\nВ предоставленном контексте отсутствует информация\n-"

        prompt = self.prompt_builder.build(query, context)
        raw = self.llm.generate(prompt)

        return self._postprocess(raw)

    def _postprocess(self, text: str) -> str:
        text = text.replace("КОНЕЦ_ОТВЕТА", "").strip()

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        unique_lines = list(dict.fromkeys(lines))

        if len(unique_lines) < 3:
            unique_lines += ["-"] * (3 - len(unique_lines))

        return "\n".join(unique_lines[:5])