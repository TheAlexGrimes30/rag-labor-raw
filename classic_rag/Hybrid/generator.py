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
        Ты — эксперт по трудовому праву Российской Федерации.

        Твоя задача — дать ТОЧНЫЙ и ПОЛНЫЙ ответ строго на основе контекста.

        КРИТИЧЕСКИЕ ПРАВИЛА:
        - НЕ сокращай формулировки закона
        - ЕСЛИ в контексте есть список — ОБЯЗАТЕЛЬНО приведи его полностью
        - НЕ обобщай (нельзя писать "и другие")
        - Используй формулировки максимально близкие к тексту закона
        - Если есть несколько пунктов — оформи их списком

        ФОРМАТ:
        Ответ:
        <полный развёрнутый ответ>

        Источник:
        <статья ТК РФ или "-">

        =====================
        ПРИМЕР:

        КОНТЕКСТ:
        Сферы правового регулирования:
        Трудовое законодательство регулирует отношения, связанные с:
        - организацией труда
        - управлением трудом
        - трудоустройством у данного работодателя

        ВОПРОС:
        что регулирует трудовое законодательство

        Ответ:
        Трудовое законодательство регулирует отношения, связанные с:
        - организацией труда
        - управлением трудом
        - трудоустройством у данного работодателя

        Источник:
        Статья 1 ТК РФ

        =====================

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

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()