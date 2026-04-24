from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    @lru_cache(maxsize=1)
    def _model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
