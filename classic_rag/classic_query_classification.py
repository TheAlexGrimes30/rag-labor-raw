import re
from typing import Literal


QueryType = Literal[
    "article_lookup",
    "concept_search",
    "case_analysis",
    "procedure",
    "legality_check",
    "general_question"
]


def classify_query(query: str) -> QueryType:
    """
    Query Classification.

    This function determines the intent of a user query
    using rule-based pattern matching.

    Args:
        query (str): Raw user query

    Returns:
        Literal[str]:
            One of predefined query types:
                - article_lookup → поиск статьи закона
                - concept_search → поиск определения/понятия
                - case_analysis → анализ ситуации пользователя
                - procedure → как что-то сделать
                - legality_check → проверка законности
                - general_question → общий вопрос
    """

    query = query.lower().strip()

    article_patterns = [
        r"(статья|ст\.)\s*\d+",
        r"\b\d+\s*(тк\s*рф)\b",
        r"\bст\s*\d+\b"
    ]

    if any(re.search(pattern, query) for pattern in article_patterns):
        return "article_lookup"

    legality_patterns = [
        "законно ли",
        "имеет ли право",
        "может ли работодатель",
        "нарушение ли",
        "правомерно ли"
    ]

    if any(p in query for p in legality_patterns):
        return "legality_check"

    procedure_patterns = [
        "как",
        "каким образом",
        "что нужно чтобы",
        "порядок",
        "процедура"
    ]

    if any(p in query for p in procedure_patterns):
        return "procedure"


    concept_patterns = [
        "что такое",
        "понятие",
        "определение",
        "что означает"
    ]

    concept_keywords = [
        "трудовой договор",
        "социальное партнерство",
        "трудовые отношения",
        "работник",
        "работодатель",
        "трудовые права"
    ]

    if any(p in query for p in concept_patterns):
        return "concept_search"

    if any(k in query for k in concept_keywords):
        return "concept_search"

    case_patterns = [
        "меня",
        "мне",
        "работодатель",
        "уволили",
        "не платят",
        "задерживают зарплату",
        "в моей ситуации"
    ]

    if any(p in query for p in case_patterns):
        return "case_analysis"

    return "general_question"


if __name__ == "__main__":
    queries = [
        "Ст. 1 ТК РФ",
        "Что такое трудовой договор?",
        "Какие права у работника?",
        "Меня уволили без причины",
        "Законно ли удержание зарплаты?",
        "Как оформить отпуск?"
    ]

    for q in queries:
        print(f"{q} -> {classify_query(q)}")
