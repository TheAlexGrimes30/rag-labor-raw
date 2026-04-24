import hashlib
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from classic_rag.Hybrid.rag_config import SearchResult
from classic_rag.Hybrid.storage import VectorStore


class BaseDenseRetriever(ABC):
    """
    Абстрактный интерфейс для dense retrieval (векторный поиск).

    Используется для:
    - поиска по эмбеддингам
    - работы с векторными БД (Qdrant, FAISS и т.д.)
    """

    @abstractmethod
    def search(self, query_vec: List[float], k: int) -> List[SearchResult]:
        """
        Выполнить поиск по вектору запроса.

        Args:
            query_vec (List[float]): Вектор запроса
            k (int): Количество возвращаемых результатов

        Returns:
            List[SearchResult]: Список найденных документов
        """

        raise NotImplementedError

class BaseSparseRetriever(ABC):
    """
    Абстрактный интерфейс для sparse retrieval (лексический поиск).
    """

    @abstractmethod
    def search(self, query: str, corpus: List[str], k: int) -> List[float]:
        """
        Выполнить лексический поиск.

        Args:
            query (str): Текст запроса
            corpus (List[str]): Список документов (тексты)
            k (int): Количество документов (используется для совместимости)

        Returns:
            List[float]: Список score для каждого документа корпуса
        """

        raise NotImplementedError

class BaseRetriever(ABC):
    """
    Абстрактный интерфейс для полного retriever.

    Объединяет:
    - dense retrieval
    - sparse retrieval
    - fusion
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """
        Выполнить поиск документов по запросу.

        Args:
            query (str): Запрос пользователя
            k (int): Количество результатов

        Returns:
            List[SearchResult]: Список релевантных документов
        """

        raise NotImplementedError

class BaseFusion(ABC):
    """
    Абстрактный интерфейс для объединения (fusion) результатов.
    """

    @abstractmethod
    def fuse(
        self,
        dense: List[SearchResult],
        sparse_scores: List[float],
    ) -> List[SearchResult]:
        """
        Объединить dense и sparse результаты.

        Args:
            dense (List[SearchResult]): Результаты dense поиска
            sparse_scores (List[float]): Оценки BM25 для тех же документов

        Returns:
            List[SearchResult]: Итоговый список документов
        """

        raise NotImplementedError

@dataclass
class Embedder:
    """
    Единый слой эмбеддингов.

    Отвечает за:
    - загрузку модели SentenceTransformer
    - добавление префиксов (E5: query:/passage:)
    - batch encoding
    - нормализацию векторов

    Attributes:
        model_name (str): Название модели эмбеддингов
        batch_size (int): Размер батча
        normalize (bool): Нормализовать ли вектора
    """

    model_name: str
    batch_size: int = 16
    normalize: bool = True

    def __post_init__(self):
        self._model = self._load_model(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        """
        Загрузка модели с кешированием.

        Args:
            model_name (str): Название модели

        Returns:
            SentenceTransformer: Загруженная модель
        """

        return SentenceTransformer(model_name)

    def encode_queries(self, texts: List[str]) -> List[List[float]]:
        """
        Закодировать список запросов.

        Args:
            texts (List[str]): Список текстов

        Returns:
            List[List[float]]: Список векторов
        """
        print("QUERY:", texts[:1])
        return self._encode(self._apply_prefix(texts, is_query=True))

    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        """
        Закодировать документы (passages).

        Args:
            texts (List[str]): Список текстов

        Returns:
            List[List[float]]: Список векторов
        """

        print("PASSAGE:", texts[:1])
        return self._encode(self._apply_prefix(texts, is_query=False))

    def encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Универсальный encode метод.

        Args:
            texts (List[str]): Тексты
            is_query (bool): Является ли текст запросом

        Returns:
            List[List[float]]: Вектора
        """

        return self._encode(self._apply_prefix(texts, is_query))

    def _apply_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        """
        Добавление E5-префиксов.

        Args:
            texts (List[str]): Тексты
            is_query (bool): Тип текста

        Returns:
            List[str]: Префиксированные тексты
        """

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            return [prefix + t for t in texts]
        return texts

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """
        Преобразование текста в вектора.

        Args:
            texts (List[str]): Список текстов

        Returns:
            List[List[float]]: Список векторов
        """

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return vectors.tolist()


class BM25Retriever(BaseSparseRetriever):
    """
    Реализация sparse retrieval через BM25.

    Attributes:
        _bm25 (Optional[BM25Okapi]): Модель BM25
        _corpus (Optional[List[str]]): Текущий корпус
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: Optional[List[str]] = None

    def build(self, corpus: List[str]) -> None:
        """
        Построение BM25 индекса.

        Args:
            corpus (List[str]): Список документов
        """

        tokenized = [self._tokenize(t) for t in corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus = corpus

    def search(self, query: str, corpus: List[str], k: int) -> List[float]:
        """
        Получить BM25 оценки для документов.

        Args:
            query (str): Запрос
            corpus (List[str]): Документы
            k (int): Не используется (для совместимости)

        Returns:
            List[float]: Нормализованные оценки
        """

        if self._bm25 is None or self._corpus != corpus:
            self.build(corpus)

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        return self._minmax(scores)

    def _tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста.

        Args:
            text (str): Входной текст

        Returns:
            List[str]: Список токенов
        """

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def _minmax(self, values: List[float]) -> List[float]:
        """
        Min-max нормализация.

        Args:
            values (List[float]): Список значений

        Returns:
            List[float]: Нормализованные значения
        """

        mn, mx = min(values), max(values)
        if abs(mx - mn) < 1e-8:
            return [0.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

class QdrantDenseRetriever(BaseDenseRetriever):
    """
    Dense retriever на основе Qdrant.

    Attributes:
        vector_store (VectorStore): Векторное хранилище
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query_vec: List[float], k: int) -> List[SearchResult]:
        """
        Поиск ближайших векторов.

        Args:
            query_vec (List[float]): Вектор запроса
            k (int): Количество результатов

        Returns:
            List[SearchResult]: Результаты поиска
        """

        hits = self.vector_store.search(query_vec, limit=k)

        results = []
        for h in hits:
            sr = SearchResult.from_qdrant(h)
            if sr.text:
                results.append(sr)

        return results

class AlphaFusion(BaseFusion):
    """
    Простая линейная комбинация dense + sparse.

    score = alpha * dense + (1 - alpha) * sparse
    """

    def __init__(self, alpha: float = 0.7, min_score: float = 0.05):
        self.alpha = alpha
        self.min_score = min_score

    def fuse(
        self,
        dense: List[SearchResult],
        sparse_scores: List[float],
    ) -> List[SearchResult]:
        """
        Объединение результатов.

        Args:
            dense (List[SearchResult]): Dense результаты
            sparse_scores (List[float]): BM25 оценки

        Returns:
            List[SearchResult]: Отсортированный список
        """

        if not dense:
            return []

        if len(sparse_scores) < len(dense):
            sparse_scores = sparse_scores + [0.0] * (len(dense) - len(sparse_scores))

        dense_scores = self._minmax([d.score for d in dense])

        fused = []
        seen = set()

        for i, doc in enumerate(dense):
            score = self.alpha * dense_scores[i] + (1 - self.alpha) * sparse_scores[i]

            if score < self.min_score:
                continue

            key = hashlib.md5((doc.text or "").encode()).hexdigest()

            if key in seen:
                continue

            seen.add(key)

            fused.append(
                SearchResult(
                    text=doc.text,
                    score=score,
                    payload=doc.payload,
                    id=doc.id,
                    source=doc.source,
                )
            )

        return sorted(fused, key=lambda x: x.score, reverse=True)

    def _minmax(self, values: List[float]) -> List[float]:
        """
        Нормализация значений.

        Args:
            values (List[float]): Значения

        Returns:
            List[float]: Нормализованные значения
        """

        mn, mx = min(values), max(values)
        if abs(mx - mn) < 1e-8:
            return [0.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

class Retriever:
    """
    Гибридный retriever:
    - Dense (Qdrant)
    - Sparse (BM25)
    - Fusion (Alpha)

    Attributes:
        vector_store (VectorStore)
        embedder (Embedder)
        dense (BaseDenseRetriever)
        sparse (BaseSparseRetriever)
        fusion (BaseFusion)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        dense: BaseDenseRetriever | None = None,
        sparse: BaseSparseRetriever | None = None,
        fusion: BaseFusion | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

        self.dense = dense or QdrantDenseRetriever(vector_store)
        self.sparse = sparse or BM25Retriever()
        self.fusion = fusion or AlphaFusion()

        self._corpus: List[str] = []
        self._is_built = False

    def build_corpus(self):
        """
        Забираем ВСЕ документы из Qdrant payload'ов
        и строим BM25 индекс.
        """

        if self._is_built:
            return

        points, _ = self.vector_store.client.scroll(
            collection_name=self.vector_store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        self._corpus = [
            (p.payload or {}).get("text", "").strip()
            for p in points
        ]

        self._corpus = [t for t in self._corpus if t]

        self.sparse.build(self._corpus)
        self._is_built = True

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Выполнить гибридный поиск.

        Алгоритм:
        1. Кодируем запрос
        2. Dense retrieval
        3. BM25 scoring
        4. Fusion
        5. Возвращаем top_k

        Args:
            query (str): Запрос пользователя
            top_k (int): Количество результатов

        Returns:
            List[SearchResult]: Итоговые документы
        """

        self.build_corpus()

        query_vec = self.embedder.encode_queries([query or ""])[0]
        dense_hits = self.dense.search(query_vec, k=top_k * 5)

        if not dense_hits:
            return []

        sparse_scores = self.sparse.search(
            query or "",
            self._corpus,
            k=len(self._corpus)
        )

        sparse_map = {
            self._corpus[i]: sparse_scores[i]
            for i in range(len(self._corpus))
        }

        aligned_sparse = [
            sparse_map.get((d.text or "").strip(), 0.0)
            for d in dense_hits
        ]

        fused = self.fusion.fuse(dense_hits, aligned_sparse)

        return [
                   f for f in fused
                   if f.text and len(f.text.strip()) > 20
               ][:top_k]
