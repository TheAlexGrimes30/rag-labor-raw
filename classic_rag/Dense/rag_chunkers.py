import re
import hashlib
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from classic_rag.Dense.rag_config import Chunk, ChunkMetadata


class Sectioner:

    def extract_sections(self, text: str) -> List[Dict]:
        sections = []
        current = {"header": None, "level": 0, "content": []}

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

    def inject(self, header: str, text: str) -> str:
        if not header:
            return text

        return f"[{header}] {text}"


class ChunkValidator:

    def __init__(self, min_chars: int = 150, min_words: int = 25):
        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(self, text: str) -> bool:
        text = text.strip()

        if not text:
            return False

        if len(text) < self.min_chars:
            return False

        if len(text.split()) < self.min_words:
            return False

        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.25:
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

            if self._is_list(buffer) != self._is_list(chunk):
                merged.append(buffer)
                buffer = chunk
                continue

            if buffer.endswith(".") and len(chunk) > 180:
                merged.append(buffer)
                buffer = chunk
                continue

            if len(buffer) + len(chunk) <= self.max_size:
                buffer += " " + chunk
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
            separators=["\n\n", "\n", ". ", "; "],
        )

        self.sectioner = Sectioner()
        self.injector = ContextInjector()
        self.validator = ChunkValidator()
        self.merger = SemanticMerger(chunk_size)

    def _extract_article(self, header, frontmatter):
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
        raw = f"{filepath}:{index}:{text[:200]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def build_metadata(self, header, frontmatter, filepath, level, article_number, chunk_index):

        return ChunkMetadata(
            source=frontmatter.get("source", "unknown"),
            file=filepath.split("/")[-1],
            header=header,
            level=level,
            article_number=article_number,
            chunk_index=chunk_index,
            topics=(frontmatter.get("classic_rag", {}) or {}).get("topics", []),
        )

    def process_section(self, section, base_metadata, filepath):

        header = section["header"]

        raw_text = "\n".join(section["content"]).strip()
        if not raw_text:
            return []

        full_text = self.injector.inject(header, raw_text)

        raw_parts = self.splitter.split_text(full_text)
        merged_parts = self.merger.merge(raw_parts)

        chunks = []

        for i, part in enumerate(merged_parts):
            part = part.strip()

            if not self.validator.is_valid(part):
                continue

            chunk_id = self._make_chunk_id(part, filepath, i)

            metadata = ChunkMetadata(
                source=base_metadata.source,
                file=base_metadata.file,
                header=base_metadata.header,
                level=base_metadata.level,
                article_number=base_metadata.article_number,
                chunk_index=i,
                topics=base_metadata.topics
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    metadata=metadata
                )
            )

        return chunks

    def create_chunks(self, sections, frontmatter, filepath):

        all_chunks = []

        for sec in sections:

            article_number = self._extract_article(sec["header"], frontmatter)

            metadata = self.build_metadata(
                header=sec["header"],
                frontmatter=frontmatter,
                filepath=filepath,
                level=sec["level"],
                article_number=article_number,
                chunk_index=0
            )

            all_chunks.extend(
                self.process_section(sec, metadata, filepath)
            )

        return all_chunks

    def process(self, filepath: str, frontmatter: dict, body: str):

        sections = self.sectioner.extract_sections(body)
        return self.create_chunks(sections, frontmatter, filepath)