from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml
from langchain_core.documents import Document

from classic_rag.Hybrid.rag_config import Chunk


class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(self) -> List[Document]:
        raise NotImplementedError

class MarkdownDocumentLoader(BaseDocumentLoader):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:
        docs: List[Document] = []

        for file_path in self.data_dir.rglob("*.md"):
            try:
                raw = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            metadata, content = self._parse_markdown(raw)

            content = (content or "").strip()
            if not content:
                continue

            classic = metadata.get("classic_rag", {}) or {}
            topics = classic.get("topics", []) if isinstance(classic, dict) else []

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file": file_path.name,
                        "topics": topics,
                        "classic_rag": classic
                    }
                )
            )

        print(f"Loaded docs: {len(docs)}")
        return docs

    def _parse_markdown(self, text: str) -> Tuple[Dict[str, Any], str]:
        if not text.startswith("---"):
            return {}, text

        try:
            parts = text.split("---", 2)
            if len(parts) < 3:
                return {}, text

            meta_raw = yaml.safe_load(parts[1]) or {}
            if not isinstance(meta_raw, dict):
                meta_raw = {}

            content = parts[2]

            classic = meta_raw.get("classic_rag", {})
            if not isinstance(classic, dict):
                classic = {}

            return {"classic_rag": classic}, content

        except Exception:
            return {}, text


class IngestionPipeline:
    def __init__(self, loader, chunker):
        self.loader = loader
        self.chunker = chunker

    def run(self) -> List[Chunk]:
        docs = self.loader.load()

        if not docs:
            return []

        chunks: List[Chunk] = []

        for doc in docs:
            text = doc.page_content

            source = doc.metadata.get("source")

            if not text or not text.strip():
                continue

            chunked = self.chunker.split(text, source=source)

            for ch in chunked:
                if not ch.text or not ch.text.strip():
                    continue

                payload = dict(doc.metadata)
                payload.update(ch.payload or {})

                new_chunk = Chunk(
                    text=ch.text.strip(),
                    payload=payload
                )

                chunks.append(new_chunk)

        return self._clean(chunks)

    def _clean(self, chunks: List[Chunk]) -> List[Chunk]:
        return [
            c for c in chunks
            if c.text and c.text.strip()
        ]