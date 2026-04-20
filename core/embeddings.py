from abc import ABC, abstractmethod
from functools import lru_cache

from config import settings


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns list of float vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = settings.openai_embedding_model):
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI recommends replacing newlines for embedding quality
        cleaned = [t.replace("\n", " ") for t in texts]
        response = self._client.embeddings.create(
            input=cleaned,
            model=self._model,
        )
        return [item.embedding for item in response.data]


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model: str = settings.hf_embedding_model):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """Singleton embedder — instantiated once, reused across requests."""
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbedder()
    return HuggingFaceEmbedder()
