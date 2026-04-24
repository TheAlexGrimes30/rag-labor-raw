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
            temperature=0.1,
            top_p=0.85,
            repeat_penalty=1.2,
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=120,
            temperature=0.05,
            stop=["=== ВОПРОС ===", "=== КОНТЕКСТ ===", "=== ОТВЕТ ==="]
        )

        answer = output["choices"][0]["text"].strip()

        answer = re.sub(r"\*\*.*?\*\*", "", answer)
        answer = re.sub(r"\n{2,}", "\n", answer)

        lines = [l.strip() for l in answer.split("\n") if l.strip()]

        clean_lines = []
        for l in lines:
            if not clean_lines or clean_lines[-1] != l:
                clean_lines.append(l)

        clean_lines = clean_lines[:3]

        return "\n".join(clean_lines)

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

                === ЗАДАЧА ===
                Сформируй короткий юридически точный ответ.

                === ФОРМАТ ОТВЕТА (ОБЯЗАТЕЛЬНО) ===
                1 строка — "Да/Нет/Краткий ответ"
                2 строка — пояснение
                3 строка — ссылка на ТК РФ (если есть)

                === ПРАВИЛА ===
                - НЕ пиши "НЕТ ОТВЕТА"
                - ВСЕГДА пытайся ответить по контексту
                - Если информации мало — дай общий юридически корректный ответ
                - НЕ рассуждай
                - НЕ повторяй вопрос
                - Убери лишние пробелы и переносы

                === ПРИМЕР ===
                Можно ли работать с 14 лет?
                Да. Можно. В соответствии со статьёй 63 ТК РФ. Несовершеннолетние с 14 лет могут работать с согласия родителей и органов опеки в свободное от учёбы время.

                === КОНТЕКСТ ===
                {context}

                === ВОПРОС ===
                {query}

                === ОТВЕТ ===
                """.strip()

class Generator(BaseGenerator):
    def __init__(
            self,
            llm: BaseLLMClient,
            prompt_builder: BasePromptBuilder,
            cleaner: ContextCleaner
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cleaner = cleaner

    def generate(self, query: str, context: str) -> str:
        context = self.cleaner.clean_context(context)

        if not context:
            return "Нет данных в источнике."

        prompt = self.prompt_builder.build(query, context)

        return self.llm.generate(prompt)

