import re
import hashlib

from chonkie import RecursiveChunker
from chonkie.refinery import OverlapRefinery

from RAG_base.chunck_generation.chunck_config import (
    ChunkMetadata,
    Chunk
)


class Sectioner:
    """
    Парсер секций Markdown-документа.

    Разбивает Markdown-документ на иерархические секции
    на основе заголовков уровней H1–H6.

    Каждая секция содержит:
    - текст заголовка
    - уровень заголовка
    - строки содержимого, расположенные под этим заголовком

    Используется как этап предварительной обработки
    перед созданием чанков.
    """

    def extract_sections(
        self,
        text: str
    ) -> list[dict]:
        """
        Извлекает секции из Markdown-документа.

        Аргументы:
            text (str):
                Исходный Markdown-документ.

        Возвращает:
            list[dict]:
                Список секций.
        """

        sections = []

        current = {
            "header": None,
            "level": 0,
            "content": []
        }

        for line in text.splitlines():

            line = line.rstrip()

            match = re.match(
                r"^(#{1,6})\s+(.+)",
                line
            )

            if match:

                if current["content"]:

                    sections.append(
                        current
                    )

                current = {
                    "header": (
                        match
                        .group(2)
                        .strip()
                    ),
                    "level": len(
                        match.group(1)
                    ),
                    "content": []
                }

            else:

                current["content"].append(
                    line
                )

        if current["content"]:

            sections.append(
                current
            )

        return sections


class ContextInjector:
    """
    Добавляет юридический контекст
    непосредственно в готовый чанк.

    Каждый чанк получает:
    - номер статьи
    - заголовок секции

    Благодаря этому каждый чанк является
    самостоятельной единицей для retrieval.
    """

    def inject(
        self,
        article_number: str | None,
        header: str | None,
        text: str
    ) -> str:

        context = []

        if article_number:

            context.append(
                f"Статья {article_number}"
            )

        if header:

            context.append(
                header
            )

        ctx = " > ".join(
            context
        )

        if not ctx:

            return text

        return (
            f"[{ctx}]\n\n"
            f"{text}"
        )


class ChunkValidator:
    """
    Проверяет качество итогового чанка.
    """

    def __init__(
        self,
        min_chars: int = 50,
        min_words: int = 7
    ):

        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(
        self,
        text: str
    ) -> bool:

        text = text.strip()

        if not text:

            return False

        if len(text) < self.min_chars:

            return False

        if (
            len(text.split())
            < self.min_words
        ):

            return False

        alpha_count = sum(
            char.isalpha()
            for char in text
        )

        alpha_ratio = (
            alpha_count
            / max(
                len(text),
                1
            )
        )

        return (
            alpha_ratio >= 0.25
        )


class HybridLegalChunker:
    """
    Гибридный chunker
    для юридических Markdown-документов.

    Пайплайн:

        Markdown
            ↓
        Sectioner
            ↓
        юридическая секция
            ↓
        нормализация текста
            ↓
        RecursiveChunker
            ↓
        структурные чанки
            ↓
        OverlapRefinery
            ↓
        ContextInjector
            ↓
        ChunkValidator
            ↓
        Chunk + ChunkMetadata

    Основной принцип:

    сначала сохраняется структура документа,
    затем Chonkie делит текст по естественным
    логическим границам.
    """

    def __init__(
        self,
        chunk_size: int = 350,
        overlap_size: int = 50
    ):

        self.splitter = RecursiveChunker(
            tokenizer="word",
            chunk_size=chunk_size,
            min_characters_per_chunk=50
        )

        self.refinery = OverlapRefinery(
            context_size=overlap_size,
            method="suffix",
            mode="recursive",
            merge=True,
            inplace=True
        )

        self.sectioner = Sectioner()

        self.injector = ContextInjector()

        self.validator = ChunkValidator(
            min_chars=50,
            min_words=7
        )

        self.global_chunk_index = 0

    def _extract_article(
        self,
        header: str | None,
        frontmatter: dict
    ) -> str | None:
        """
        Извлекает номер статьи.

        Поддерживает номера:

            1
            165
            327.1
            351.7
        """

        if header:

            match = re.search(
                r"Статья\s+"
                r"(\d+(?:\.\d+)*)",
                header
            )

            if match:

                return (
                    match.group(1)
                )

        doc_id = frontmatter.get(
            "id",
            ""
        )

        match = re.search(
            r"article_"
            r"(\d+(?:\.\d+)*)",
            doc_id
        )

        if match:

            return (
                match.group(1)
            )

        article = frontmatter.get(
            "article"
        )

        if article is None:

            return None

        return str(
            article
        )

    def _make_chunk_id(
        self,
        text: str,
        filepath: str,
        article_number: str | None,
        header: str | None
    ) -> str:
        """
        Создаёт детерминированный ID чанка.

        ID не зависит от глобального индекса,
        поэтому меньше меняется при повторной
        индексации документов.
        """

        raw = "|".join(
            [
                filepath,
                str(
                    article_number or ""
                ),
                str(
                    header or ""
                ),
                text
            ]
        )

        return hashlib.md5(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()

    def _prepare_legal_text(
        self,
        text: str
    ) -> str:
        """
        Нормализует юридический текст.

        Важно:
        переносы строк сохраняются,
        потому что они содержат структуру:

        - абзацы
        - списки
        - подпункты
        """

        text = text.replace(
            "\r\n",
            "\n"
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"^\s*---+\s*$",
            "",
            text,
            flags=re.MULTILINE
        )

        return (
            text.strip()
        )

    def _extract_chunk_text(
        self,
        chunk
    ) -> str:
        """
        Извлекает текст
        из Chonkie Chunk.
        """

        if hasattr(
            chunk,
            "text"
        ):

            return (
                chunk.text.strip()
            )

        return (
            str(chunk)
            .strip()
        )

    def process_section(
        self,
        section: dict,
        base_metadata: ChunkMetadata,
        filepath: str
    ) -> list[Chunk]:
        """
        Преобразует одну Markdown-секцию
        в набор юридических чанков.
        """

        header = (
            section["header"]
        )

        raw_text = "\n".join(
            section["content"]
        )

        raw_text = (
            raw_text.strip()
        )

        if not raw_text:

            return []

        article_number = (
            base_metadata
            .article_number
        )

        text = (
            self._prepare_legal_text(
                raw_text
            )
        )

        chonkie_chunks = (
            self.splitter.chunk(
                text
            )
        )

        if (
            len(chonkie_chunks)
            > 1
        ):

            chonkie_chunks = (
                self.refinery.refine(
                    chonkie_chunks
                )
            )

        results = []

        for chonkie_chunk in (
            chonkie_chunks
        ):

            part = (
                self._extract_chunk_text(
                    chonkie_chunk
                )
            )

            if not part:

                continue

            part = (
                self.injector.inject(
                    article_number,
                    header,
                    part
                )
            )

            if not (
                self.validator
                .is_valid(part)
            ):

                continue

            idx = (
                self.global_chunk_index
            )

            chunk_id = (
                self._make_chunk_id(
                    text=part,
                    filepath=filepath,
                    article_number=(
                        article_number
                    ),
                    header=header
                )
            )

            metadata = ChunkMetadata(
                source=(
                    base_metadata.source
                ),
                file=(
                    base_metadata.file
                ),
                header=header,
                level=(
                    base_metadata.level
                ),
                article_number=(
                    article_number
                ),
                chunk_index=idx,
                topics=(
                    base_metadata.topics
                )
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

    def create_chunks(
        self,
        sections: list[dict],
        frontmatter: dict,
        filepath: str
    ) -> list[Chunk]:
        """
        Создаёт чанки
        для всех секций документа.
        """

        all_chunks = []

        for section in sections:

            article_number = (
                self._extract_article(
                    section["header"],
                    frontmatter
                )
            )

            metadata = ChunkMetadata(
                source=frontmatter.get(
                    "source",
                    "unknown"
                ),
                file=filepath,
                header=(
                    section["header"]
                ),
                level=(
                    section["level"]
                ),
                article_number=(
                    article_number
                ),
                chunk_index=None,
                topics=(
                    (
                        frontmatter.get(
                            "classic_rag",
                            {}
                        )
                        or {}
                    )
                    .get(
                        "topics",
                        []
                    )
                )
            )

            section_chunks = (
                self.process_section(
                    section=section,
                    base_metadata=metadata,
                    filepath=filepath
                )
            )

            all_chunks.extend(
                section_chunks
            )

        return all_chunks

    def process(
        self,
        filepath: str,
        frontmatter: dict,
        body: str
    ) -> list[Chunk]:
        """
        Основная точка входа.
        """

        sections = (
            self.sectioner
            .extract_sections(
                body
            )
        )

        return (
            self.create_chunks(
                sections=sections,
                frontmatter=frontmatter,
                filepath=filepath
            )
        )