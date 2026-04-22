from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]