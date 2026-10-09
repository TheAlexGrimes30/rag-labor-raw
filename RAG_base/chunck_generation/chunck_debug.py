from pathlib import Path

from RAG_base.chunck_generation.ingestion import IngestionService, IngestionPipeline, MarkdownDocumentLoader
from RAG_base.chunck_generation.rag_chunkers import HybridLegalChunker


def save_chunks_to_file(
    chunks,
    output_path: Path
) -> None:
    """
    Сохраняет созданные чанки в текстовый файл
    для ручной проверки качества разбиения.

    Каждый чанк записывается отдельно вместе
    с его идентификатором и метаданными.

    Аргументы:
        chunks:
            Список созданных RAG-чанков.

        output_path (Path):
            Путь к итоговому тестовому файлу.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            metadata = chunk.metadata

            file.write(
                "=" * 100
            )

            file.write("\n")

            file.write(
                f"CHUNK #{index}\n"
            )

            file.write(
                "=" * 100
            )

            file.write("\n\n")

            file.write(
                f"chunk_id: {chunk.chunk_id}\n"
            )

            file.write(
                f"source: {metadata.source}\n"
            )

            file.write(
                f"file: {metadata.file}\n"
            )

            file.write(
                f"header: {metadata.header}\n"
            )

            file.write(
                f"level: {metadata.level}\n"
            )

            file.write(
                f"article_number: {metadata.article_number}\n"
            )

            file.write(
                f"chunk_index: {metadata.chunk_index}\n"
            )

            file.write(
                f"topics: {metadata.topics}\n"
            )

            file.write("\n")

            file.write(
                "-" * 100
            )

            file.write(
                "\nTEXT\n"
            )

            file.write(
                "-" * 100
            )

            file.write("\n\n")

            file.write(
                chunk.text
            )

            file.write("\n\n\n")


def main() -> None:
    """
    Тестовая точка входа для проверки
    качества созданных чанков.

    Выполняет:
    1. Определение путей проекта
    2. Создание loader
    3. Создание chunker
    4. Запуск ingestion pipeline
    5. Сохранение чанков в файл
    """

    project_root = Path(
        __file__
    ).resolve().parents[2]

    data_path = (
        project_root
        / "rag_db"
    )

    output_path = (
        project_root
        / "chunks_debug.txt"
    )

    loader = MarkdownDocumentLoader(
        data_dir=str(data_path)
    )

    chunker = HybridLegalChunker()

    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker
    )

    ingestion_service = IngestionService(
        pipeline=pipeline
    )

    chunks = (
        ingestion_service.load_chunks()
    )

    save_chunks_to_file(
        chunks=chunks,
        output_path=output_path
    )

    print(
        f"[Chunk Test] Всего чанков: {len(chunks)}"
    )

    print(
        f"[Chunk Test] Результат сохранён: {output_path}"
    )


if __name__ == "__main__":
    main()