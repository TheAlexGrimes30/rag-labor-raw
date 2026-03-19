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
    Query Classification

    :param: query: str

    Returns:
        QueryType: One of the following categories:
            - "article_lookup"   : запрос на конкретную статью закона (например: "Ст. 1 ТК РФ")
            - "concept_search"   : запрос на определение или юридическое понятие (например: "что такое трудовой договор")
            - "case_analysis"    : описание ситуации пользователя (например: "меня уволили без причины")
            - "procedure"        : запрос о том, как выполнить действие (например: "как оформить отпуск")
            - "legality_check"   : проверка законности (например: "законно ли удержание зарплаты")
            - "general_question" : общий вопрос, не попадающий в другие категории
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
