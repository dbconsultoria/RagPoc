"""Embeddings via Ollama (/api/embed)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_local.config import Settings
from rag_local.ollama import OllamaClient, OllamaError


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        dim: int,
        doc_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self.client = client
        self.model = model
        self.dim = dim
        # nomic-embed-text foi treinado com prefixos de tarefa distintos para
        # documento e pergunta; sem eles a qualidade do retrieval cai.
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaEmbedder:
        return cls(
            OllamaClient(settings.ollama_url),
            model=settings.embed_model,
            dim=settings.embed_dim,
            doc_prefix=settings.embed_doc_prefix,
            query_prefix=settings.embed_query_prefix,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors.extend(self._embed([self.doc_prefix + t for t in batch]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        data = self.client.post("/api/embed", {"model": self.model, "input": inputs})
        vectors: list[list[float]] = data["embeddings"]
        if len(vectors) != len(inputs):
            raise OllamaError(f"Ollama devolveu {len(vectors)} embeddings para {len(inputs)} textos")
        for vector in vectors:
            if len(vector) != self.dim:
                raise OllamaError(
                    f"O modelo '{self.model}' gera vetores de {len(vector)} dimensões, "
                    f"mas embed_dim está configurado como {self.dim}"
                )
        return vectors
