from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:
        documents: List[Document] = []

        for file_path in self.data_dir.rglob("*.md"):
            try:
                raw = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            metadata, content = self._parse_markdown(raw)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file": file_path.name,
                        "topics": metadata.get("classic_rag", {}).get("topics", [])
                    }
                )
            )

        return documents

    def _parse_markdown(self, text: str) -> Tuple[Dict[str, Any], str]:

        if not text.startswith("---"):
            return {}, text

        try:
            parts = text.split("---", 2)

            if len(parts) < 3:
                return {}, text

            meta_raw = yaml.safe_load(parts[1]) or {}
            content = parts[2].strip()

            classic = meta_raw.get("classic_rag", {})

            return {
                "classic_rag": classic
            }, content

        except Exception:
            return {}, text


class IngestionPipeline:
    def __init__(self, loader, chunker):
        self.loader = loader
        self.chunker = chunker

    def run(self) -> List[Document]:
        docs = self.loader.load()

        if not docs:
            return []

        chunks = []

        for doc in docs:
            text = doc.page_content
            source = doc.metadata.get("source")

            chunked = self.chunker.split(text, source=source)

            for ch in chunked:
                chunks.append(
                    Document(
                        page_content=ch.text,
                        metadata={
                            **doc.metadata,
                            **ch.payload
                        }
                    )
                )

        return self._clean(chunks)

    def _clean(self, chunks: List[Document]) -> List[Document]:
        return [
            c for c in chunks
            if c.page_content and c.page_content.strip()
        ]
    