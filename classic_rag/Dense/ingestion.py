import re
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
from langchain_core.documents import Document

from classic_rag.Dense.rag_config import Chunk


class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Defines the common interface for loading documents
    from any source (Markdown, PDF, database, API, etc.).
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """
        Load documents from a source.

        Returns:
            List[Document]:
                List of LangChain Document objects.
        """

        raise NotImplementedError

class BasePipeline(ABC):
    """
    Abstract base class for ingestion pipelines.

    Defines the interface for document processing pipelines.
    """

    @abstractmethod
    def run(self) -> list[Chunk]:
        """
        Execute pipeline processing.

        Returns:
            List[Chunk]:
                List of processed chunks.
        """

        raise NotImplementedError

class BaseIngestionService(ABC):
    """
    Abstract base class for ingestion services.

    Defines high-level ingestion operations.
    """

    @abstractmethod
    def load_chunks(self) -> list[Chunk]:
        """
        Load processed chunks from pipeline.

        Returns:
            List[Chunk]:
                List of generated chunks.
        """

        raise NotImplementedError


class MarkdownDocumentLoader(BaseDocumentLoader):
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
        """
        Load all markdown file paths from directory recursively.

        Returns:
            list[Path]:
                List of markdown file paths.
        """

        return list(self.data_dir.rglob("*.md"))

    def parse_file(self, path: str) -> tuple[dict, str]:
        """
        Parse markdown file into frontmatter metadata and body.

        Supports YAML frontmatter in format:

            key: value
            markdown content

        Args:
            path (str):
                Path to markdown file.

        Returns:
            Tuple[Dict, str]:
                Tuple containing:
                    - parsed YAML frontmatter metadata
                    - markdown body text
        """

        text = Path(path).read_text(encoding="utf-8")

        match = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)

        if match:
            frontmatter = yaml.safe_load(match.group(1)) or {}
            body = match.group(2)
        else:
            frontmatter = {}
            body = text

        return frontmatter, body

class IngestionPipeline(BasePipeline):
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

class IngestionService(BaseIngestionService):
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