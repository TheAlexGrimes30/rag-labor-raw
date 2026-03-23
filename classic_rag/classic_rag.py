import re
import string
from typing import Literal

from langchain_community.embeddings import HuggingFaceEmbeddings

def process_query(query: str) -> str:
    """
    Query Processing
    """

    legal_abbreviations = {
        r"\bст\.\b": "статья",
        r"\bтк\s*рф\b": "трудовой кодекс рф",
        r"\bгк\s*рф\b": "гражданский кодекс рф",
        r"\bнк\s*рф\b": "налоговый кодекс рф",
        r"\bроструд\b": "федеральная служба по труду и занятости",
        r"\bфнс\b": "налоговая служба",
        r"\bвс\s*рф\b": "верховный суд рф",
    }

    query = query.strip().lower()

    for abbr, full in legal_abbreviations.items():
        query = re.sub(abbr, full, query)

    query = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)
    query = re.sub(r"\s+", " ", query)

    return query.strip()

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

    if any(p in query for p in concept_patterns):
        return "concept_search"

    return "general_question"

def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

def prepare_documents(docs):
    for doc in docs:
        doc.page_content = "passage: " + doc.page_content.strip()
    return docs