from abc import abstractmethod, ABC
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from classic_rag.Hybrid.rag_config import Chunk


class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(self) -> List[Document]:
        raise NotImplementedError

class MarkdownDocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[Path]:
        files = list(self.data_dir.rglob("*.md"))
        print(f"[Loader] Found files: {len(files)}")
        return files

class IngestionPipeline:
    def __init__(self, loader, chunker):
        self.loader = loader
        self.chunker = chunker

    def run(self) -> List[Chunk]:
        files = self.loader.load()

        if not files:
            return []

        chunks: List[Chunk] = []

        for path in files:
            file_chunks = self.chunker.process(str(path))
            chunks.extend(file_chunks)

        chunks = [c for c in chunks if (c.text or "").strip()]


        return chunks