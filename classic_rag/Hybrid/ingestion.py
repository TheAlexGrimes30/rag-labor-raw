import re
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
from langchain_core.documents import Document

from classic_rag.Hybrid.rag_config import Chunk


class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(self) -> List[Document]:
        raise NotImplementedError

class MarkdownDocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> list[Path]:
        return list(self.data_dir.rglob("*.md"))

    def parse_file(self, path: str) -> Tuple[Dict, str]:
        text = Path(path).read_text(encoding="utf-8")

        match = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)

        if match:
            frontmatter = yaml.safe_load(match.group(1)) or {}
            body = match.group(2)
        else:
            frontmatter = {}
            body = text

        return frontmatter, body

class IngestionPipeline:
    def __init__(self, loader, chunker):
        self.loader = loader
        self.chunker = chunker

    def run(self) -> List[Chunk]:
        chunks = []

        for path in self.loader.load():

            frontmatter, body = self.loader.parse_file(str(path))

            chunks.extend(
                self.chunker.process(
                    filepath=str(path),
                    frontmatter=frontmatter,
                    body=body
                )
            )

        return [c for c in chunks if c.text.strip()]
