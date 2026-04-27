import re
from abc import abstractmethod, ABC
from typing import List

from classic_rag.Hybrid.rag_config import Chunk


class BaseChunker(ABC):

    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
    YAML_PATTERN = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
    SECTION_SPLIT = re.compile(r"\n---\n")

    def _normalize(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", (text or "").strip())

    def _split_sentences(self, text: str) -> List[str]:
        return self.SENTENCE_SPLIT_REGEX.split(text)

    def _strip_yaml(self, text: str) -> str:
        """
        Удаляет YAML metadata в начале markdown файла
        """
        return self.YAML_PATTERN.sub("", text).strip()

    def _split_sections(self, text: str) -> List[str]:
        """
        Делит текст по --- (логические блоки)
        """
        parts = self.SECTION_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    @abstractmethod
    def split(self, text: str, *, source: str | None = None) -> List[Chunk]:
        pass

class SentenceChunker(BaseChunker):

    def __init__(self, chunk_size: int = 800, overlap_sentences: int = 2):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def split(self, text: str, *, source: str | None = None) -> List[Chunk]:

        text = self._normalize(text)
        if not text:
            return []

        sentences = self._split_sentences(text)

        chunks: List[Chunk] = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            if current_len + sent_len > self.chunk_size:

                chunk_text = " ".join(current).strip()

                if len(chunk_text) > 120:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            payload={
                                "source": source,
                                "chunk_type": "sentence"
                            }
                        )
                    )

                current = current[-self.overlap_sentences:]
                current_len = sum(len(x) for x in current)

            current.append(sent)
            current_len += sent_len

        if current:
            chunk_text = " ".join(current).strip()

            if len(chunk_text) > 120:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        payload={
                            "source": source,
                            "chunk_type": "sentence"
                        }
                    )
                )

        return chunks


class WindowChunker(BaseChunker):

    def __init__(self, max_chars: int = 800, overlap: int = 150):
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, text: str, *, source: str | None = None):

        text = self._normalize(text)
        if not text:
            return []

        chunks = []
        step = max(1, self.max_chars - self.overlap)

        for start in range(0, len(text), step):
            end = min(start + self.max_chars, len(text))
            piece = text[start:end].strip()

            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        payload={
                            "source": source,
                            "chunk_type": "window",
                            "start": start,
                            "end": end,
                        }
                    )
                )

        return chunks

class StructureChunker(BaseChunker):

    HEADER_PATTERN = re.compile(r"^(#{1,4})\s+(.+)", re.MULTILINE)
    ARTICLE_PATTERN = re.compile(r"Статья\s+(\d+)")

    def __init__(self, fallback_chunker: BaseChunker | None = None):
        self.fallback = fallback_chunker

    def split(self, text: str, *, source: str | None = None):

        text = self._strip_yaml(text)
        text = self._normalize(text)

        if not text:
            return []

        sections = self._split_sections(text)
        all_chunks = []

        for section in sections:
            article_match = self.ARTICLE_PATTERN.search(section)
            if article_match:
                all_chunks.append(
                    Chunk(
                        text=section.strip(),
                        payload={
                            "source": source,
                            "chunk_type": "article",
                            "article_number": article_match.group(1)
                        }
                    )
                )
                continue

            matches = list(self.HEADER_PATTERN.finditer(section))

            if not matches:
                if self.fallback:
                    all_chunks.extend(self.fallback.split(section, source=source))
                continue

            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(section)

                header = match.group(2).strip()
                level = len(match.group(1))

                body = section[start:end].strip()

                payload = {
                    "source": source,
                    "chunk_type": "structure",
                    "header": header,
                    "level": level,
                }

                all_chunks.append(
                    Chunk(
                        text=body,
                        payload=payload
                    )
                )

        return all_chunks


class SmartChunker(BaseChunker):

    def __init__(
        self,
        structure_chunker: StructureChunker,
        sentence_chunker: SentenceChunker,
        window_chunker: WindowChunker,
        max_chars: int = 800
    ):
        self.structure = structure_chunker
        self.sentence = sentence_chunker
        self.window = window_chunker
        self.max_chars = max_chars

    def split(self, text: str, *, source: str | None = None):

        structured = self.structure.split(text, source=source)

        final_chunks = []

        for chunk in structured:

            if len(chunk.text) <= self.max_chars:
                final_chunks.append(chunk)
                continue

            sentence_chunks = self.sentence.split(chunk.text, source=source)

            if sentence_chunks:
                for sc in sentence_chunks:
                    sc.payload.update({
                        "parent_header": chunk.payload.get("header"),
                        "parent_level": chunk.payload.get("level"),
                        "parent_type": "structure"
                    })
                    final_chunks.append(sc)
                continue

            window_chunks = self.window.split(chunk.text, source=source)

            for wc in window_chunks:
                wc.payload.update({
                    "parent_header": chunk.payload.get("header"),
                    "parent_level": chunk.payload.get("level"),
                    "parent_type": "structure"
                })
                final_chunks.append(wc)

        return final_chunks
