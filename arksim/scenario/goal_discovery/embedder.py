# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingService(ABC):
    """Common interface for embedding services."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts and return an (N, D) float32 array."""


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Local embedding via sentence-transformers (no API calls, no cost).

    Requires: pip install sentence-transformers
    Note: depends on torch + transformers; may conflict in some environments.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers\n"
                "Or use embedding_provider='openai' to avoid this dependency."
            ) from e

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Embedding via OpenAI API (text-embedding-3-small by default).

    Requires OPENAI_API_KEY. No local model or torch dependency.
    Costs ~$0.02 per 1M tokens, negligible for typical corpus sizes.
    """

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        """Embed texts in batches and return an (N, D) float32 array.

        OpenAI normalises embeddings by default for text-embedding-3-* models.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                input=batch, model=self.model_name
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return np.array(all_embeddings, dtype=np.float32)


# Convenience alias kept for backward compatibility
EmbeddingService = SentenceTransformerEmbeddingService


def build_embedding_service(
    provider: str = "openai",
    model_name: str | None = None,
) -> BaseEmbeddingService:
    """Factory that returns the right embedding service by provider name.

    Args:
        provider: "openai" or "sentence-transformers".
        model_name: Override the default model for the chosen provider.
    """
    if provider == "openai":
        return OpenAIEmbeddingService(model_name or "text-embedding-3-small")
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingService(model_name or "all-MiniLM-L6-v2")
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Use 'openai' or 'sentence-transformers'."
    )
