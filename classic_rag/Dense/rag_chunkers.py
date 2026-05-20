import re
import hashlib

from chonkie import SentenceChunker
from chonkie.refinery import OverlapRefinery

from classic_rag.Dense.rag_config import ChunkMetadata, Chunk


class Sectioner:
    """
    Markdown section parser.

    Splits a Markdown document into hierarchical sections based on headers (H1–H6).
    Each section contains:
    - header text
    - header level
    - content lines under this header

    This is used as a preprocessing step before chunking.
    """

    def extract_sections(self, text: str) -> list[dict]:
        """
        Extract sections from a Markdown document.

        Args:
            text (str): Raw Markdown document.

        Returns:
            List[Dict]: List of sections with structure:
                {
                    "header": str | None,
                    "level": int,
                    "content": List[str]
                }
        """

        sections = []
        current = {
            "header": None,
            "level": 0,
            "content": []
        }

        for line in text.split("\n"):

            line = line.rstrip()

            if not line.strip():
                continue

            match = re.match(r'^(#{1,6})\s+(.+)', line)

            if match:

                if current["content"]:
                    sections.append(current)

                current = {
                    "header": match.group(2).strip(),
                    "level": len(match.group(1)),
                    "content": []
                }

            else:
                current["content"].append(line)

        if current["content"]:
            sections.append(current)

        return sections


class ContextInjector:
    """
    Injects legal context into text.

    Adds structured metadata (article number + section header)
    into the chunk text to improve retrieval and reranking quality.
    """

    def inject(self, article_number: str, header: str, text: str) -> str:
        """
        Inject legal context into raw text.

        Args:
            article_number (str): Legal article number (e.g., "307")
            header (str): Section header
            text (str): Raw section text

        Returns:
            str: Context-enhanced text
        """

        context = []

        if article_number:
            context.append(f"Статья {article_number}")

        if header:
            context.append(header)

        ctx = " > ".join(context)

        return f"[{ctx}]\n\n{text}" if ctx else text


class ChunkValidator:
    """
    Validates chunk quality.

    Filters out:
    - too short chunks
    - low-information or noisy text
    """

    def __init__(self, min_chars=120, min_words=20):
        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(self, text: str) -> bool:
        """
        Validate whether a chunk is useful for RAG.

        Args:
            text (str): Input chunk text

        Returns:
            bool: True if chunk is valid, False otherwise
        """

        text = text.strip()

        if len(text) < self.min_chars:
            return False

        if len(text.split()) < self.min_words:
            return False

        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

        return alpha_ratio >= 0.25


class HybridLegalChunker:
    """
    Hybrid legal document chunking pipeline.

    Pipeline steps:
    1. Markdown → structured sections
    2. Context injection (legal structure enrichment)
    3. Sentence-based chunking (Chonkie SentenceChunker)
    4. Overlap refinement (OverlapRefinery)
    5. Validation
    6. Metadata attachment
    """

    def __init__(self):

        self.splitter = SentenceChunker(
            chunk_size=8,
            chunk_overlap=1
        )

        self.refinery = OverlapRefinery()

        self.sectioner = Sectioner()
        self.injector = ContextInjector()
        self.validator = ChunkValidator()

        self.global_chunk_index = 0

    def _extract_article(self, header: str, frontmatter: dict) -> str | None:
        """
        Extract legal article number from header or metadata.

        Args:
            header (str): Section header
            frontmatter (dict): YAML metadata

        Returns:
            str | None: Extracted article number if available
        """

        if header:
            m = re.search(r'Статья\s+(\d+)', header)
            if m:
                return m.group(1)

        doc_id = frontmatter.get("id", "")
        m = re.search(r'article_(\d+)', doc_id)
        if m:
            return m.group(1)

        return frontmatter.get("article")

    def _make_chunk_id(self, text: str, filepath: str, index: int) -> str:
        """
        Generate deterministic chunk ID using hash.

        Args:
            text (str): Chunk text
            filepath (str): Source file path
            index (int): Global chunk index

        Returns:
            str: Unique chunk identifier
        """

        raw = f"{filepath}:{index}:{text[:200]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _prepare_legal_text(self, text: str) -> str:
        """
        Normalize legal text for better chunking.

        Args:
            text (str): Raw text

        Returns:
            str: Cleaned and normalized text
        """

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'---+', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def process_section(self,
                        section: dict,
                        base_metadata: ChunkMetadata,
                        filepath: str
                        ) -> list[Chunk]:

        """
        Convert a single section into RAG chunks.

        Args:
            section (Dict): Parsed Markdown section
            base_metadata (ChunkMetadata): Shared metadata
            filepath (str): Source file path

        Returns:
            List[Chunk]: Generated chunks
        """

        header = section["header"]
        raw_text = "\n".join(section["content"]).strip()

        if not raw_text:
            return []

        article_number = base_metadata.article_number

        text = self.injector.inject(article_number, header, raw_text)
        text = self._prepare_legal_text(text)

        chunks = self.splitter.chunk(text)

        chunks = self.refinery.refine(chunks)

        results = []

        for ch in chunks:

            part = ch.text if hasattr(ch, "text") else str(ch)
            part = part.strip()

            if not self.validator.is_valid(part):
                continue

            idx = self.global_chunk_index

            chunk_id = self._make_chunk_id(part, filepath, idx)

            metadata = ChunkMetadata(
                source=base_metadata.source,
                file=base_metadata.file,
                header=header,
                level=base_metadata.level,
                article_number=base_metadata.article_number,
                chunk_index=idx,
                topics=base_metadata.topics
            )

            results.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    metadata=metadata
                )
            )

            self.global_chunk_index += 1

        return results

    def create_chunks(self,
                      sections: list[dict],
                      frontmatter: dict,
                      filepath: str
                      ) -> list[Chunk]:
        """
        Build all chunks from parsed document sections.

        Args:
            sections (List[Dict]): Markdown sections
            frontmatter (dict): Document metadata (YAML)
            filepath (str): Source file path

        Returns:
            List[Chunk]: Final chunk list
        """


        all_chunks = []

        for sec in sections:

            article_number = self._extract_article(sec["header"], frontmatter)

            metadata = ChunkMetadata(
                source=frontmatter.get("source", "unknown"),
                file=filepath,
                header=sec["header"],
                level=sec["level"],
                article_number=article_number,
                chunk_index=self.global_chunk_index,
                topics=(frontmatter.get("classic_rag", {}) or {}).get("topics", [])
            )

            all_chunks.extend(
                self.process_section(sec, metadata, filepath)
            )

        return all_chunks

    def process(self, filepath: str, frontmatter: dict, body: str) -> list[Chunk]:
        """
        Entry point for the chunking pipeline.

        Args:
            filepath (str): Path to Markdown file
            frontmatter (dict): YAML metadata
            body (str): Raw Markdown content

        Returns:
            List[Chunk]: Final processed chunks
        """

        sections = self.sectioner.extract_sections(body)
        return self.create_chunks(sections, frontmatter, filepath)