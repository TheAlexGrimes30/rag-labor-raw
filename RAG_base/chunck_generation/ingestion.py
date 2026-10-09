import re
from pathlib import Path
from typing import List

import yaml

from RAG_base.chunck_generation.chunck_config import Chunk

class MarkdownDocumentLoader:
    """
    Loader for Markdown documents.

    Responsibilities:
    - recursively scan directory for `.md` files
    - read markdown content
    - parse YAML frontmatter metadata
    - separate metadata from document body
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> list[Path]:
        return list(self.data_dir.rglob("*.md"))

    def parse_file(self, path: str) -> tuple[dict, str]:

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
    """
    Main ingestion pipeline.

    Responsibilities:
    - load source documents
    - parse markdown files
    - send documents into chunker
    - collect all generated chunks
    """

    def __init__(self, loader, chunker):
        self.loader = loader
        self.chunker = chunker

    def run(self) -> list[Chunk]:
        """
        Execute ingestion pipeline.

        Processing steps:
        1. Load markdown files
        2. Parse frontmatter + body
        3. Chunk documents
        4. Filter empty chunks

        Returns:
            List[Chunk]:
                List of processed chunks.
        """

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

class IngestionService:
    """
    High-level ingestion service.

    Wrapper around ingestion pipeline used by the application layer.
    """

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    def load_chunks(self) -> list[Chunk]:
        """
        Load and process chunks through pipeline.

        Returns:
            List[Chunk]:
                List of generated chunks.
        """

        chunks = self.pipeline.run()
        print(f"[Ingestion] Loaded chunks: {len(chunks)}")
        return chunks