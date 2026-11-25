"""Fixtures compartilhadas. O FakeEmbedder substitui o Ollama nos testes."""

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from rag_local.config import Settings
from rag_local.store import Store

DIM = 768


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"\w+", text)


class FakeEmbedder:
    """Bag-of-words com hashing: determinístico e textos com palavras em comum ficam próximos."""

    dim = DIM

    def __init__(self) -> None:
        self.documents_embedded = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokens(text):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        if not any(vector):
            vector[0] = 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        docs_dir=tmp_path / "docs",
        db_path=tmp_path / "test.duckdb",
        chunk_size=300,
        chunk_overlap=50,
    )


@pytest.fixture
def store(settings: Settings) -> Iterator[Store]:
    with Store(settings.db_path, settings.embed_dim) as s:
        yield s


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
