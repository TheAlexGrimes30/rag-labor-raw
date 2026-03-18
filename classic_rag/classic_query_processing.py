import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def process_query(query: str) -> str:
    """
    Query Processing
    """

    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('russian'))

    legal_terms = {"статья", "часть", "пункт", "трудовой", "кодекс", "рф"}
    legal_abbreviations = {
        r"\bst\.\b": "статья",
        r"\bч\.\b": "часть",
        r"\bп\.\b": "пункт",
        r"\bтк рф\b": "трудовой кодекс рф"
    }

    stemmer = SnowballStemmer("russian")

    query = query.strip().lower()

    for abbr, full in legal_abbreviations.items():
        query = re.sub(abbr, full, query)

    query = re.sub(r"[*_#>-]", " ", query)
    query = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)

    tokens = query.split()

    processed_tokens: List[str] = []
    for token in tokens:
        if token in legal_terms:
            processed_tokens.append(token)
            continue
        if token in stop_words:
            continue
        processed_tokens.append(stemmer.stem(token))

    processed_query = " ".join(processed_tokens)

    return processed_query


if __name__ == "__main__":
    user_input = "Ст. 1 ТК РФ: Какие цели трудового законодательства?"
    processed = process_query(user_input)
    print("Исходный запрос:", user_input)
    print("Обработанный запрос:", processed)
