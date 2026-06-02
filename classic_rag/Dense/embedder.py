from functools import lru_cache

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around SentenceTransformer models.

    Responsible for:
    - loading embedding model
    - query/document encoding
    - E5 prefix handling
    - vector normalization
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self._model = self._load_model(model_name)

        self.dim = (
            self._model.get_sentence_embedding_dimension()
        )

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        """
        Load and cache embedding model.

        Args:
            model_name (str):
                HuggingFace model name.

        Returns:
            SentenceTransformer:
                Loaded embedding model.
        """

        return SentenceTransformer(model_name)

    def encode_queries(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode search queries.

        Args:
            texts (List[str]):
                Query texts.

        Returns:
            List[List[float]]:
                Query embeddings.
        """


        texts = self._apply_prefix(
            texts,
            is_query=True
        )

        return self._encode(texts)

    def encode_passages(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode document passages.

        Args:
            texts (List[str]):
                Passage texts.

        Returns:
            List[List[float]]:
                Passage embeddings.
        """

        texts = self._apply_prefix(
            texts,
            is_query=False
        )

        return self._encode(texts)

    def _apply_prefix(
        self,
        texts: list[str],
        is_query: bool
    ) -> list[str]:
        """
        Apply E5 prefixes if model requires them.

        E5 models require:
        - "query: " for queries
        - "passage: " for documents

        Args:
            texts (List[str]):
                Input texts.

            is_query (bool):
                Whether texts are queries.

         Returns:
            List[str]:
                Prefixed texts.
        """

        if "e5" not in self.model_name.lower():
            return texts

        prefix = (
            "query: "
            if is_query
            else "passage: "
        )

        return [
            prefix + t
            for t in texts
        ]

    def _encode(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts (List[str]):
                Input texts.

        Returns:
            List[List[float]]:
                Embedding vectors.
        """

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )

        return vectors.tolist()
