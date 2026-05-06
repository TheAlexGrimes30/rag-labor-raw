import re
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from classic_rag.Hybrid.rag_config import Chunk, ChunkMetadata

class Sectioner:

    def extract_sections(self, text: str) -> List[Dict]:
        sections = []
        current = {"header": None, "level": 0, "content": []}

        for line in text.split("\n"):
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
    def inject(self, header: str, text: str) -> str:
        if not header:
            return text

        return f"{header}\n\n{text}"

class ChunkValidator:
    def __init__(self, min_chars: int = 80, min_alpha_ratio: float = 0.3):
        self.min_chars = min_chars
        self.min_alpha_ratio = min_alpha_ratio

    def is_valid(self, text: str) -> bool:
        text = text.strip()

        if len(text) < self.min_chars:
            return False

        if not text:
            return False

        alpha = sum(c.isalpha() for c in text)
        if alpha / max(len(text), 1) < self.min_alpha_ratio:
            return False

        if len(text.split("\n")) == 1 and len(text) < 120:
            return False

        return True

class SemanticMerger:
    def __init__(self, max_size: int = 900):
        self.max_size = max_size

    def _is_list(self, text: str) -> bool:
        return bool(re.match(r"^\s*[-•*]\s+", text))

    def merge(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        merged = []
        buffer = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if not buffer:
                buffer = chunk
                continue

            should_merge = (
                len(buffer) + len(chunk) <= self.max_size
                and (
                    self._is_list(buffer)
                    or self._is_list(chunk)
                    or buffer.endswith(":")
                    or len(chunk) < 100
                )
            )

            if should_merge:
                buffer += "\n" + chunk
            else:
                merged.append(buffer)
                buffer = chunk

        if buffer:
            merged.append(buffer)

        return merged

class HybridLegalChunker:

    def __init__(self, chunk_size=800, chunk_overlap=120):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n#### ",
                "\n### ",
                "\n## ",
                "\n\n",
                "\n",
                " "
            ],
        )

        self.sectioner = Sectioner()
        self.injector = ContextInjector()
        self.validator = ChunkValidator()
        self.merger = SemanticMerger(chunk_size)

        self.chunk_size = chunk_size

    def _extract_article(self, header, frontmatter):
        if header:
            m = re.search(r'Статья\s+(\d+)', header)
            if m:
                return m.group(1)

        doc_id = frontmatter.get("id", "")
        m = re.search(r'article_(\d+)', doc_id)
        return m.group(1) if m else None

    def build_metadata(self, header, frontmatter, filepath):

        return ChunkMetadata(
            source=frontmatter.get("source", "unknown"),
            file=filepath.split("/")[-1],
            header=header,
            level=0,
            article_number=self._extract_article(header, frontmatter),
            topics=(frontmatter.get("classic_rag", {}) or {}).get("topics", [])
        )

    def process_section(self, section, metadata):

        header = section["header"]
        raw_text = "\n".join(section["content"]).strip()

        if not raw_text:
            return []

        full_text = self.injector.inject(header, raw_text)

        raw_parts = self.splitter.split_text(full_text)

        merged_parts = self.merger.merge(raw_parts)

        chunks = []

        for part in merged_parts:
            part = part.strip()

            if not self.validator.is_valid(part):
                continue

            chunks.append(
                Chunk(
                    text=part,
                    metadata=metadata
                )
            )

        return chunks

    def create_chunks(self, sections, frontmatter, filepath):

        all_chunks = []

        for sec in sections:
            metadata = self.build_metadata(sec["header"], frontmatter, filepath)

            chunks = self.process_section(sec, metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def process(self, filepath: str, frontmatter: dict, body: str):
        sections = self.sectioner.extract_sections(body)
        return self.create_chunks(sections, frontmatter, filepath)

