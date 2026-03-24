import re
from typing import List

from classic_rag.classic_query_classification import classify_query
from classic_rag.classic_query_processing import process_query


def translate_query(
    raw_query: str,
    processed_query: str,
    query_type: str
) -> str:
    """
    Query Translation

    Hybrid query construction:
        Combines:
            - normalized query (processed_query)
            - expanded tokens (synonyms, legal terms)
            - cleaned raw query (natural language signal)

    This improves both:
        - sparse retrieval (BM25)
        - dense retrieval (embeddings)

    Returns:
        str:
            Retrieval-optimized hybrid query
    """

    LEGAL_SYNONYMS = {
        "увол": ["увольнение", "расторжение", "прекращение"],
        "зарплат": ["заработная плата", "оплата труда"],
        "работник": ["сотрудник"],
        "работодател": ["наниматель"],
        "договор": ["трудовой договор"],
        "отпуск": ["ежегодный отпуск"],
    }

    LEGAL_BOOST_TERMS = {
        "увольнение",
        "договор",
        "работник",
        "работодатель",
        "зарплата",
        "отпуск"
    }

    def expand_synonyms(tokens: List[str]) -> List[str]:
        """Add domain-specific synonyms."""

        expanded = []

        for token in tokens:
            expanded.append(token)

            for key in LEGAL_SYNONYMS:
                if token.startswith(key):
                    expanded.extend(LEGAL_SYNONYMS[key])

        return expanded

    def boost_terms(tokens: List[str]) -> List[str]:
        """Boost important legal terms."""

        boosted = []

        for token in tokens:
            boosted.append(token)

            if token in LEGAL_BOOST_TERMS:
                boosted.append(token)

        return boosted

    def deduplicate(tokens: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""

        seen = set()
        result = []

        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)

        return result

    tokens = processed_query.split()

    if query_type == "article_lookup":
        return processed_query

    if query_type == "concept_search":
        tokens = expand_synonyms(tokens)

    elif query_type == "case_analysis":
        tokens = boost_terms(expand_synonyms(tokens))

    elif query_type == "legality_check":
        tokens = expand_synonyms(tokens)
        tokens.extend(["законность", "нарушение"])
        tokens = boost_terms(tokens)

    elif query_type == "procedure":
        tokens.extend(["порядок", "процедура"])
        tokens = expand_synonyms(tokens)

    else:
        tokens = expand_synonyms(tokens)

    raw_clean = re.sub(r"[^\w\s]", " ", raw_query.lower())
    raw_tokens = raw_clean.split()

    tokens = deduplicate(tokens)

    processed_part = processed_query
    expanded_part = " ".join(tokens)
    raw_part = " ".join(raw_tokens)

    final_query = (
        processed_part + " " +
        expanded_part + " " +
        raw_part + " " +
        raw_part
    )

    return final_query.strip()


if __name__ == "__main__":

    queries = [
        "Ст. 1 ТК РФ",
        "Что такое трудовой договор?",
        "Меня уволили без причины",
        "Законно ли не платить зарплату?",
        "Как оформить отпуск?"
    ]

    for q in queries:
        q_type = classify_query(q)
        processed = process_query(q)
        translated = translate_query(q, processed, q_type)

        print("\nQuery:", q)
        print("Type:", q_type)
        print("Processed:", processed)
        print("Translated:", translated)