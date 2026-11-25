"""Retrieval: busca vetorial, BM25 e fusão por Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_local.config import Settings
from rag_local.embeddings import Embedder
from rag_local.store import Store, StoredChunk

SearchMode = Literal["hybrid", "vector", "bm25"]


def format_citation(source: str, page: int | None) -> str:
    return f"{source}, p. {page}" if page is not None else source


@dataclass(frozen=True)
class SearchHit:
    chunk: StoredChunk
    score: float  # RRF no modo híbrido; cosseno no vetorial; BM25 no textual
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None

    @property
    def citation(self) -> str:
        return format_citation(self.chunk.source, self.chunk.page)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """score(d) = soma de 1 / (k + posição de d) em cada ranking onde d aparece.

    Usa só a posição, não o score bruto: cosseno (0..1) e BM25 (sem teto) não são
    comparáveis, e o RRF dispensa normalizá-los.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, *, candidates: int = 20, rrf_k: int = 60) -> None:
        self.store = store
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k

    @classmethod
    def from_settings(cls, store: Store, embedder: Embedder, settings: Settings) -> Retriever:
        return cls(store, embedder, candidates=settings.candidates, rrf_k=settings.rrf_k)

    def search(self, query: str, k: int = 4, mode: SearchMode = "hybrid") -> list[SearchHit]:
        limit = max(self.candidates, k)
        vector_hits: list[tuple[int, float]] = []
        bm25_hits: list[tuple[int, float]] = []
        if mode in ("vector", "hybrid"):
            vector_hits = self.store.vector_search(self.embedder.embed_query(query), limit)
        if mode in ("bm25", "hybrid"):
            bm25_hits = self.store.bm25_search(query, limit)

        if mode == "vector":
            ranked = vector_hits[:k]
        elif mode == "bm25":
            ranked = bm25_hits[:k]
        else:
            ranked = reciprocal_rank_fusion(
                [[i for i, _ in vector_hits], [i for i, _ in bm25_hits]], k=self.rrf_k
            )[:k]

        vector_info = {doc_id: (rank, score) for rank, (doc_id, score) in enumerate(vector_hits, 1)}
        bm25_info = {doc_id: (rank, score) for rank, (doc_id, score) in enumerate(bm25_hits, 1)}
        chunks = self.store.get_chunks([doc_id for doc_id, _ in ranked])

        hits: list[SearchHit] = []
        for doc_id, score in ranked:
            v_rank, v_score = vector_info.get(doc_id, (None, None))
            b_rank, b_score = bm25_info.get(doc_id, (None, None))
            hits.append(SearchHit(chunks[doc_id], score, v_rank, v_score, b_rank, b_score))
        return hits
