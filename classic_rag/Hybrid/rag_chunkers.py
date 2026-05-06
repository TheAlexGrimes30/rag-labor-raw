import re
from abc import abstractmethod, ABC
from typing import List

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from classic_rag.Hybrid.rag_config import Chunk, ChunkMetadata


class SimpleMarkdownParser:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n### ", "\n## ", "\n\n", "\n", " ", ""],
        )

    def parse_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)

        if match:
            frontmatter = yaml.safe_load(match.group(1)) or {}
            body = match.group(2)
        else:
            frontmatter = {}
            body = content

        return frontmatter, body

    def extract_sections(self, text: str):
        sections = []
        current = {"header": None, "level": 0, "content": []}

        for line in text.split("\n"):
            match = re.match(r'^(#+)\s+(.+)', line)

            if match:
                if current["content"]:
                    sections.append(current)

                current = {
                    "header": match.group(2),
                    "level": len(match.group(1)),
                    "content": []
                }
            else:
                current["content"].append(line)

        if current["content"]:
            sections.append(current)

        return sections

    def extract_article_number(self, header: str, frontmatter: dict):
        if header:
            m = re.search(r'Статья\s+(\d+)', header)
            if m:
                return m.group(1)

        doc_id = frontmatter.get("id", "")
        m = re.search(r'article_(\d+)', doc_id)
        return m.group(1) if m else None

    def create_chunks(self, sections, frontmatter):
        chunks = []

        for sec in sections:
            text = "\n".join(sec["content"]).strip()
            if not text:
                continue

            article_number = self.extract_article_number(sec["header"], frontmatter)

            base_metadata = ChunkMetadata(
                source=frontmatter.get("source", "unknown"),
                file=frontmatter.get("file", "unknown.md"),
                header=sec["header"],
                level=sec["level"],
                article_number=article_number,
                topics=frontmatter.get("classic_rag", {}).get("topics", [])
            )

            if len(text) <= self.chunk_size:
                chunks.append(Chunk(text=text, metadata=base_metadata))
            else:
                for part in self.splitter.split_text(text):
                    chunks.append(Chunk(text=part, metadata=base_metadata))

        return chunks

    def process(self, filepath: str):
        fm, body = self.parse_file(filepath)
        sections = self.extract_sections(body)
        return self.create_chunks(sections, fm)
