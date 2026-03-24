import re
from typing import List, Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from classic_rag.classic_query_classification import classify_query
from classic_rag.classic_query_processing import process_query
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def llm_expand_fn(query: str) -> str:
    """
    Expands a query semantically using LLM.
    """

    prompt = f"Перефразируй и расширь запрос для поиска юридической информации: {query}"
    result = llm_pipe(
        prompt,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True
    )
    return result[0]['generated_text']

def translate_query_hybrid(
    raw_query: str,
    processed_query: str,
    query_type: str,
    llm_expand_fn: Optional[Callable] = None
) -> str:
    """
    Hybrid Query Translation (Rules + LLM)

    1. Rules-based expansion:
        - synonyms
        - term boosting
        - type-aware augmentation
    2. LLM-assisted semantic expansion (optional)
    3. Combine processed + expanded + LLM tokens + raw query

    Args:
        raw_query (str): original user input
        processed_query (str): normalized query (from process_query)
        query_type (str): intent label (from classify_query)
        llm_expand_fn (callable): function that takes a string and returns
                                  semantically enriched query (optional)

    Returns:
        str: retrieval-optimized hybrid query
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
        expanded = []
        for token in tokens:
            expanded.append(token)
            for key in LEGAL_SYNONYMS:
                if token.startswith(key):
                    expanded.extend(LEGAL_SYNONYMS[key])
        return expanded

    def boost_terms(tokens: List[str]) -> List[str]:
        boosted = []
        for token in tokens:
            boosted.append(token)
            if token in LEGAL_BOOST_TERMS:
                boosted.append(token)
        return boosted

    def deduplicate(tokens: List[str]) -> List[str]:
        seen = set()
        result = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    tokens = processed_query.split()

    if query_type == "article_lookup":
        rule_tokens = tokens
    elif query_type == "concept_search":
        rule_tokens = expand_synonyms(tokens)
    elif query_type == "case_analysis":
        rule_tokens = boost_terms(expand_synonyms(tokens))
    elif query_type == "legality_check":
        rule_tokens = boost_terms(expand_synonyms(tokens) + ["законность", "нарушение"])
    elif query_type == "procedure":
        rule_tokens = expand_synonyms(tokens + ["порядок", "процедура"])
    else:
        rule_tokens = expand_synonyms(tokens)

    rule_tokens = deduplicate(rule_tokens)

    raw_clean = re.sub(r"[^\w\s]", " ", raw_query.lower())
    raw_tokens = raw_clean.split()

    if llm_expand_fn:
        llm_text = llm_expand_fn(raw_query)
        llm_tokens = re.sub(r"[^\w\s]", " ", llm_text.lower()).split()
    else:
        llm_tokens = []

    all_tokens = deduplicate(rule_tokens + raw_tokens + llm_tokens)
    final_query = " ".join(all_tokens)
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
        translated = translate_query_hybrid(q, processed, q_type, llm_expand_fn=llm_expand_fn)

        print("\nQuery:", q)
        print("Type:", q_type)
        print("Processed:", processed)
        print("Translated (Hybrid):", translated)
