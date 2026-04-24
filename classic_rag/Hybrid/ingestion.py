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

            content = (content or "").strip()
            if not content:
                continue

            classic = metadata.get("classic_rag", {}) or {}
            topics = classic.get("topics", []) if isinstance(classic, dict) else []

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file": file_path.name,
                        "topics": topics
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

    def run(self) -> List[Document]:
        docs = self.loader.load()

        if not docs:
            return []

        chunks: List[Document] = []

        for doc in docs:
            text = doc.page_content
            source = doc.metadata.get("source")

            if not text or not text.strip():
                continue

            chunked = self.chunker.split(text, source=source)

            for ch in chunked:
                if not ch.text or not ch.text.strip():
                    continue

                metadata = dict(doc.metadata)
                metadata.update(ch.payload or {})

                chunks.append(
                    Document(
                        page_content=ch.text.strip(),
                        metadata=metadata
                    )
                )

        return self._clean(chunks)

    def _clean(self, chunks: List[Document]) -> List[Document]:
        return [
            c for c in chunks
            if c.page_content and c.page_content.strip()
        ]