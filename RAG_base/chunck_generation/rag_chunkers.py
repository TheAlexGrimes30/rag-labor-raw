import re
import hashlib

from chonkie import SentenceChunker
from chonkie.refinery import OverlapRefinery

from RAG_base.chunck_generation.chunck_config import ChunkMetadata, Chunk


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

    def extract_sections(self, text: str) -> list[dict]:
        """
        Извлекает секции из Markdown-документа.

        Аргументы:
            text (str):
                Исходный Markdown-документ.

        Возвращает:
            list[dict]:
                Список секций со структурой:

                {
                    "header": str | None,
                    "level": int,
                    "content": list[str]
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

            match = re.match(
                r'^(#{1,6})\s+(.+)',
                line
            )

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
    Добавляет юридический контекст в текст.

    Вставляет структурированные метаданные
    (номер статьи и заголовок секции)
    непосредственно в текст чанка.

    Это позволяет улучшить качество
    поиска и последующего reranking.
    """

    def inject(
        self,
        article_number: str,
        header: str,
        text: str
    ) -> str:
        """
        Добавляет юридический контекст в исходный текст.

        Аргументы:
            article_number (str):
                Номер юридической статьи
                (например, "307").

            header (str):
                Заголовок секции.

            text (str):
                Исходный текст секции.

        Возвращает:
            str:
                Текст с добавленным юридическим контекстом.
        """

        context = []

        if article_number:
            context.append(
                f"Статья {article_number}"
            )

        if header:
            context.append(header)

        ctx = " > ".join(context)

        return (
            f"[{ctx}]\n\n{text}"
            if ctx
            else text
        )


class ChunkValidator:
    """
    Проверяет качество чанка.

    Отфильтровывает:
    - слишком короткие чанки
    - малоинформативный или шумный текст
    """

    def __init__(
        self,
        min_chars=120,
        min_words=20
    ):
        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(self, text: str) -> bool:
        """
        Проверяет, подходит ли чанк для использования в RAG.

        Аргументы:
            text (str):
                Текст проверяемого чанка.

        Возвращает:
            bool:
                True, если чанк является допустимым,
                иначе False.
        """

        text = text.strip()

        if len(text) < self.min_chars:
            return False

        if len(text.split()) < self.min_words:
            return False

        alpha_ratio = (
            sum(
                c.isalpha()
                for c in text
            )
            / max(len(text), 1)
        )

        return alpha_ratio >= 0.25


class HybridLegalChunker:
    """
    Гибридный пайплайн создания чанков
    для юридических документов.

    Этапы пайплайна:
    1. Markdown → структурированные секции
    2. Добавление контекста
       (обогащение юридической структурой)
    3. Разбиение по предложениям
       (Chonkie SentenceChunker)
    4. Добавление перекрытия
       между чанками (OverlapRefinery)
    5. Валидация чанков
    6. Добавление метаданных
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

    def _extract_article(
        self,
        header: str,
        frontmatter: dict
    ) -> str | None:
        """
        Извлекает номер юридической статьи
        из заголовка или метаданных документа.

        Аргументы:
            header (str):
                Заголовок секции.

            frontmatter (dict):
                YAML-метаданные документа.

        Возвращает:
            str | None:
                Найденный номер статьи,
                если он присутствует.
        """

        if header:
            m = re.search(
                r'Статья\s+(\d+)',
                header
            )

            if m:
                return m.group(1)

        doc_id = frontmatter.get(
            "id",
            ""
        )

        m = re.search(
            r'article_(\d+)',
            doc_id
        )

        if m:
            return m.group(1)

        return frontmatter.get("article")

    def _make_chunk_id(
        self,
        text: str,
        filepath: str,
        index: int
    ) -> str:
        """
        Создаёт детерминированный идентификатор чанка
        с использованием хеширования.

        Аргументы:
            text (str):
                Текст чанка.

            filepath (str):
                Путь к исходному файлу.

            index (int):
                Глобальный индекс чанка.

        Возвращает:
            str:
                Уникальный идентификатор чанка.
        """

        raw = (
            f"{filepath}:"
            f"{index}:"
            f"{text[:200]}"
        )

        return hashlib.md5(
            raw.encode("utf-8")
        ).hexdigest()

    def _prepare_legal_text(
        self,
        text: str
    ) -> str:
        """
        Нормализует юридический текст
        перед созданием чанков.

        Аргументы:
            text (str):
                Исходный текст.

        Возвращает:
            str:
                Очищенный и нормализованный текст.
        """

        text = re.sub(
            r'\n{3,}',
            '\n\n',
            text
        )

        text = re.sub(
            r'---+',
            '',
            text
        )

        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        return text.strip()

    def process_section(
        self,
        section: dict,
        base_metadata: ChunkMetadata,
        filepath: str
    ) -> list[Chunk]:
        """
        Преобразует одну секцию документа
        в набор RAG-чанков.

        Аргументы:
            section (dict):
                Разобранная Markdown-секция.

            base_metadata (ChunkMetadata):
                Общие метаданные секции.

            filepath (str):
                Путь к исходному файлу.

        Возвращает:
            list[Chunk]:
                Список созданных чанков.
        """

        header = section["header"]

        raw_text = "\n".join(
            section["content"]
        ).strip()

        if not raw_text:
            return []

        article_number = (
            base_metadata.article_number
        )

        text = self.injector.inject(
            article_number,
            header,
            raw_text
        )

        text = self._prepare_legal_text(
            text
        )

        chunks = self.splitter.chunk(
            text
        )

        chunks = self.refinery.refine(
            chunks
        )

        results = []

        for ch in chunks:

            part = (
                ch.text
                if hasattr(ch, "text")
                else str(ch)
            )

            part = part.strip()

            if not self.validator.is_valid(
                part
            ):
                continue

            idx = self.global_chunk_index

            chunk_id = self._make_chunk_id(
                part,
                filepath,
                idx
            )

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

    def create_chunks(
        self,
        sections: list[dict],
        frontmatter: dict,
        filepath: str
    ) -> list[Chunk]:
        """
        Создаёт все чанки
        из разобранных секций документа.

        Аргументы:
            sections (list[dict]):
                Список Markdown-секций.

            frontmatter (dict):
                YAML-метаданные документа.

            filepath (str):
                Путь к исходному файлу.

        Возвращает:
            list[Chunk]:
                Итоговый список чанков.
        """

        all_chunks = []

        for sec in sections:

            article_number = (
                self._extract_article(
                    sec["header"],
                    frontmatter
                )
            )

            metadata = ChunkMetadata(
                source=frontmatter.get(
                    "source",
                    "unknown"
                ),
                file=filepath,
                header=sec["header"],
                level=sec["level"],
                article_number=article_number,
                chunk_index=self.global_chunk_index,
                topics=(
                    frontmatter.get(
                        "classic_rag",
                        {}
                    )
                    or {}
                ).get(
                    "topics",
                    []
                )
            )

            all_chunks.extend(
                self.process_section(
                    sec,
                    metadata,
                    filepath
                )
            )

        return all_chunks

    def process(
        self,
        filepath: str,
        frontmatter: dict,
        body: str
    ) -> list[Chunk]:
        """
        Основная точка входа
        в пайплайн создания чанков.

        Аргументы:
            filepath (str):
                Путь к Markdown-файлу.

            frontmatter (dict):
                YAML-метаданные документа.

            body (str):
                Исходное содержимое Markdown-документа.

        Возвращает:
            list[Chunk]:
                Итоговый список обработанных чанков.
        """

        sections = (
            self.sectioner.extract_sections(
                body
            )
        )

        return self.create_chunks(
            sections,
            frontmatter,
            filepath
        )
