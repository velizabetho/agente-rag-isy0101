"""Text embeddings: local sentence-transformers (default) or OpenAI."""

from functools import lru_cache
from typing import Protocol

from bookclub.config import get_settings
from bookclub.observability.recorder import span


class Embedder(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """sentence-transformers model. E5 models expect 'query: ' / 'passage: ' prefixes."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._e5 = "e5" in model_name.lower()
        self.dimensions = self._model.get_embedding_dimension()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with span("encode", kind="embedding") as s:
            s["texts"] = len(texts)
            vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {t}" if self._e5 else t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}" if self._e5 else text])[0]


class OpenAIEmbedder:
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        self.dimensions = 1536 if model_name.endswith("small") else 3072

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.embedding_model, settings.openai_api_key)
    with span("load_embedding_model", kind="embedding"):  # cold start: only the first call of the process
        return LocalEmbedder(settings.embedding_model)
