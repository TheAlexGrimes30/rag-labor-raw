import re
from pathlib import Path

from llama_cpp import Llama


class Generator:

    def __init__(self):
        print("Loading LLM...")

        base_dir = Path(__file__).resolve().parent.parent.parent
        model_path = base_dir / "models" / "Qwen3-8B-Q4_K_M.gguf"

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            temperature=0.1,
            top_p=0.85,
            repeat_penalty=1.2,
        )

    def clean_context(self, text: str):
        text = re.sub(r"#+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def build_prompt(self, query: str, context: str):
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

    def generate(self, query: str, context: str):
        context = self.clean_context(context)

        if not context:
            return "Нет данных в источнике."

        prompt = self.build_prompt(query, context)

        output = self.llm(
            prompt,
            max_tokens=220,
            temperature=0.1,
            stop=["=== ВОПРОС ===", "=== КОНТЕКСТ ==="]
        )

        answer = output["choices"][0]["text"].strip()
        answer = re.sub(r"\n{2,}", "\n", answer)
        answer = answer.strip()

        return answer