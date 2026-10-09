import re
from chonkie import RecursiveChunker
from chonkie.refinery import OverlapRefinery

from RAG_base.chunck_generation.chunck_config import (
    ChunkMetadata,
    Chunk
)

class Sectioner:
    """Разделяет Markdown на секции и сохраняет путь родительских заголовков."""

    _HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

    def extract_sections(self, text: str) -> list[dict]:
        sections: list[dict] = []
        headings: list[tuple[int, str]] = []
        current = {"header": None, "level": None, "content": [], "heading_path": []}
        in_fence = False
        fence_char = None
        fence_len = 0

        for line in text.splitlines():
            stripped = line.strip()
            fence = re.match(r"^(`{3,}|~{3,})", stripped)

            if fence:
                marker = fence.group(1)

                if not in_fence:
                    in_fence, fence_char, fence_len = True, marker[0], len(marker)

                elif marker[0] == fence_char and len(marker) >= fence_len:
                    in_fence, fence_char, fence_len = False, None, 0

                current["content"].append(line.rstrip())
                continue

            match = None if in_fence else self._HEADING_PATTERN.match(line)

            if match:
                if any(part.strip() for part in current["content"]):
                    sections.append(current)

                level = len(match.group(1))
                header = match.group(2).strip()

                while headings and headings[-1][0] >= level:
                    headings.pop()

                headings.append((level, header))

                current = {
                    "header": header,
                    "level": level,
                    "content": [],
                    "heading_path": [name for _, name in headings],
                }

            else:
                current["content"].append(line.rstrip())

        if any(part.strip() for part in current["content"]):
            sections.append(current)
        return sections


class ContextInjector:
    """Добавляет название акта, номер статьи и путь заголовков перед чанком."""

    def inject(
        self,
        article_number: str | None,
        header: str | None,
        text: str,
        source: str | None = None,
        heading_path: list[str] | None = None,
    ) -> str:

        context: list[str] = []

        if source and source.strip() and source.strip().lower() != "unknown":
            context.append(source.strip())

        path = heading_path if heading_path is not None else ([header] if header else [])

        article_heading = any(
            re.match(r"^Статья\s+\d+(?:\.\d+)*\b", item, re.IGNORECASE)
            for item in path
        )

        if article_number and not article_heading:
            context.append(f"Статья {article_number}")

        context.extend(item.strip() for item in path if item and item.strip())

        if not context:
            return text

        return f"[{' > '.join(context)}]\n\n{text}"


class ChunkValidator:
    """Проверяет исходное содержимое чанка до добавления контекста."""

    def __init__(self, min_chars: int = 50, min_words: int = 7):
        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(self, text: str) -> bool:
        text = text.strip()

        if not text or len(text) < self.min_chars:
            return False

        if len(text.split()) < self.min_words:
            return False

        alpha_count = sum(char.isalpha() for char in text)

        return alpha_count / len(text) >= 0.25


class HybridLegalChunker:
    """Markdown -> секции -> Chonkie -> overlap -> validation -> context -> Chunk.
    """

    _ARTICLE_IN_HEADER = re.compile(
        r"\bСтатья\s+(\d+(?:\.\d+)*(?:-\d+)?)\b",
        re.IGNORECASE
    )

    _ARTICLE_IN_ID = re.compile(
        r"(?:^|_)article_(\d+(?:_\d+)*)(?=_|$)",
        re.IGNORECASE
    )

    def __init__(
        self,
        chunk_size: int = 350,
        overlap_size: int = 50,
        min_chars: int = 50,
        min_words: int = 7,
    ):
        if chunk_size <= 0 or overlap_size < 0:
            raise ValueError("chunk_size должен быть > 0, overlap_size должен быть >= 0")

        self.splitter = RecursiveChunker(
            tokenizer="word",
            chunk_size=chunk_size,
            min_characters_per_chunk=min_chars,
        )

        self.refinery = OverlapRefinery(
            context_size=overlap_size,
            method="suffix",
            mode="recursive",
            merge=True,
            inplace=True,
        ) if overlap_size else None

        self.sectioner = Sectioner()
        self.injector = ContextInjector()
        self.validator = ChunkValidator(min_chars=min_chars, min_words=min_words)

    def _extract_article(
            self,
            header: str | None,
            frontmatter: dict,
    ) -> str | None:

        yaml_article = frontmatter.get("article")
        yaml_number = (
            str(yaml_article).strip()
            if yaml_article is not None
            else None
        )

        header_match = self._ARTICLE_IN_HEADER.search(header or "")
        header_number = header_match.group(1) if header_match else None

        doc_id = str(frontmatter.get("id") or "")
        id_match = self._ARTICLE_IN_ID.search(doc_id)

        id_number = (
            id_match.group(1).replace("_", ".")
            if id_match
            else None
        )

        # YAML — основной источник номера статьи.
        if yaml_number:
            if header_number and yaml_number != header_number:
                raise ValueError(
                    f"Несовпадение номера статьи: "
                    f"YAML={yaml_number}, header={header_number}"
                )

            return yaml_number

        # Если YAML отсутствует, используем заголовок.
        if header_number:
            return header_number

        # Последний резервный источник — ID.
        return id_number

    def _prepare_legal_text(self, text: str) -> str:

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+(?=\n)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_chunk_text(self, chunk) -> str:
        return (chunk.text if hasattr(chunk, "text") else str(chunk)).strip()

    def process_section(
        self,
        section: dict,
        base_metadata: ChunkMetadata,
        filepath: str,
        start_index: int = 0,
    ) -> list[Chunk]:
        """Обрабатывает секцию, сохраняя формат существующих Chunk/ChunkMetadata."""

        header = section.get("header")
        raw_text = "\n".join(section.get("content", [])).strip()

        if not raw_text:
            return []

        text = self._prepare_legal_text(raw_text)
        parts = self.splitter.chunk(text)

        if self.refinery is not None and len(parts) > 1:
            parts = self.refinery.refine(parts)

        results: list[Chunk] = []
        for part_obj in parts:
            part = self._extract_chunk_text(part_obj)
            if not self.validator.is_valid(part):
                continue

            contextual_text = self.injector.inject(
                article_number=base_metadata.article_number,
                header=header,
                text=part,
                source=base_metadata.source,
                heading_path=section.get("heading_path"),
            )

            metadata = ChunkMetadata(
                source=base_metadata.source,
                file=base_metadata.file,
                header=header,
                level=section.get("level"),
                article_number=base_metadata.article_number,
                chunk_index=start_index + len(results),
                topics=list(base_metadata.topics),
            )

            results.append(Chunk(text=contextual_text, metadata=metadata))
        return results

    def create_chunks(
        self,
        sections: list[dict],
        frontmatter: dict,
        filepath: str,
    ) -> list[Chunk]:
        """Собирает чанки документа с нумерацией от нуля."""

        all_chunks: list[Chunk] = []
        classic_rag = frontmatter.get("classic_rag") or {}
        topics = classic_rag.get("topics") or []

        for section in sections:
            article_number = self._extract_article(section.get("header"), frontmatter)

            metadata = ChunkMetadata(
                source=str(frontmatter.get("source") or frontmatter.get("law") or "unknown"),
                file=filepath,
                header=section.get("header"),
                level=section.get("level"),
                article_number=article_number,
                chunk_index=None,
                topics=list(topics),
            )

            all_chunks.extend(
                self.process_section(
                    section=section,
                    base_metadata=metadata,
                    filepath=filepath,
                    start_index=len(all_chunks),
                )
            )
        return all_chunks

    def process(self, filepath: str, frontmatter: dict, body: str) -> list[Chunk]:
        sections = self.sectioner.extract_sections(body)
        return self.create_chunks(sections, frontmatter, filepath)
