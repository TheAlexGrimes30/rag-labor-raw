import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Dict

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

        scores = self._bm25.get_scores(self._tokenize(query))
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

    def __init__(self, alpha: float = 0.7, min_score: float = 0.05):
        self.alpha = alpha
        self.min_score = min_score

    def fuse(
        self,
        dense: List[SearchResult],
        sparse_dict: Dict[str, float],
    ) -> List[SearchResult]:

        if not dense:
            return []

        dense_scores = self._minmax([d.score for d in dense])

        fused = []

        for i, doc in enumerate(dense):

            sparse_score = sparse_dict.get(str(doc.id), 0.0)

            score = self.alpha * dense_scores[i] + (1 - self.alpha) * sparse_score

            if score < self.min_score:
                continue

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
        mn, mx = min(values), max(values)
        if abs(mx - mn) < 1e-8:
            return [0.0] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

class Retriever:

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

        self.dense = QdrantDenseRetriever(vector_store)
        self.sparse = BM25Retriever()
        self.fusion = AlphaFusion(alpha=0.7)

        self._corpus: List[str] = []
        self._id_map: Dict[str, int] = {}
        self._reverse_id_map: Dict[int, str] = {}
        self._is_built = False

    def build_corpus(self):

        if self._is_built:
            return

        points, _ = self.vector_store.client.scroll(
            collection_name=self.vector_store.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        corpus = []
        id_map = {}
        reverse_map = {}

        for p in points:
            payload = p.payload or {}
            text = (payload.get("text") or "").strip()

            if not text:
                continue

            idx = len(corpus)
            corpus.append(text)

            doc_id = str(p.id)

            id_map[doc_id] = idx
            reverse_map[idx] = doc_id

        self._corpus = corpus
        self._id_map = id_map
        self._reverse_id_map = reverse_map

        self.sparse.build(self._corpus)
        self._is_built = True

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:

        self.build_corpus()

        query_vec = self.embedder.encode_queries([query])[0]

        dense_hits = self.dense.search(query_vec, k=top_k * 5)

        if not dense_hits:
            return []

        sparse_scores = self.sparse.search(query, self._corpus, k=len(self._corpus))

        sparse_dict = {
            self._reverse_id_map[i]: score
            for i, score in enumerate(sparse_scores)
        }

        for d in dense_hits:
            if str(d.id) not in sparse_dict:
                try:
                    idx = self._corpus.index(d.text)
                    sparse_dict[str(d.id)] = sparse_scores[idx]
                except ValueError:
                    sparse_dict[str(d.id)] = 0.0

        fused = self.fusion.fuse(dense_hits, sparse_dict)

        return fused[:top_k]

    def debug_query(self, query: str, top_k: int = 10) -> None:
        print("\n" + "=" * 80)
        print(f"[QUERY]: {query}")

        self.build_corpus()

        query_vec = self.embedder.encode_queries([query])[0]

        dense_hits = self.dense.search(query_vec, k=top_k * 5)

        print(f"\n[DENSE TOP {top_k}]")
        for i, d in enumerate(dense_hits[:top_k]):
            print(f"{i + 1}. score={d.score:.4f} | id={d.id}")
            print(f"   text: {d.text[:120]}...\n")

        sparse_scores = self.sparse.search(query, self._corpus, k=len(self._corpus))

        top_sparse_idx = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i],
            reverse=True
        )[:top_k]

        print(f"\n[BM25 TOP {top_k}]")
        for rank, idx in enumerate(top_sparse_idx):
            doc_id = self._reverse_id_map[idx]
            print(f"{rank + 1}. score={sparse_scores[idx]:.4f} | id={doc_id}")
            print(f"   text: {self._corpus[idx][:120]}...\n")

        # Mapping
        sparse_dict = {
            self._reverse_id_map[i]: score
            for i, score in enumerate(sparse_scores)
        }

        print(f"\n[ALIGNMENT CHECK]")
        for d in dense_hits[:top_k]:
            s = sparse_dict.get(str(d.id), None)
            print(f"id={d.id} | dense={d.score:.4f} | sparse={s}")

        fused = self.fusion.fuse(dense_hits, sparse_dict)

        print(f"\n[FUSED TOP {top_k}]")
        for i, f in enumerate(fused[:top_k]):
            print(f"{i + 1}. score={f.score:.4f} | id={f.id}")
            print(f"   text: {f.text[:120]}...\n")

        print("=" * 80)