"""Avaliação do retrieval: hit rate@k e MRR@k sobre um conjunto de perguntas com gabarito."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_local.retrieval import Retriever, SearchHit, SearchMode
from rag_local.store import StoredChunk


@dataclass(frozen=True)
class EvalCase:
    pergunta: str
    fonte: str | None = None
    pagina: int | None = None
    trecho: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    rank: int | None  # posição (1-based) do primeiro chunk relevante; None se não veio no top k
    hits: list[SearchHit]


@dataclass(frozen=True)
class EvalReport:
    mode: str
    k: int
    results: list[CaseResult]

    @property
    def hit_rate(self) -> float:
        return sum(r.rank is not None for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results) if self.results else 0.0


def normalize(text: str) -> str:
    """Minúsculas, sem acento e com espaços colapsados, para comparar trechos com robustez."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def load_cases(path: Path) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("perguntas", []) if isinstance(data, dict) else data
    cases: list[EvalCase] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not item.get("pergunta"):
            raise ValueError(f"{path.name}: item {position} precisa ter 'pergunta'")
        if not (item.get("fonte") or item.get("trecho")):
            raise ValueError(f"{path.name}: item {position} precisa de 'fonte' e/ou 'trecho'")
        unknown = set(item) - {"pergunta", "fonte", "pagina", "trecho"}
        if unknown:
            raise ValueError(f"{path.name}: item {position} tem chaves desconhecidas: {', '.join(unknown)}")
        cases.append(
            EvalCase(
                pergunta=str(item["pergunta"]),
                fonte=item.get("fonte"),
                pagina=int(item["pagina"]) if item.get("pagina") is not None else None,
                trecho=item.get("trecho"),
            )
        )
    return cases


def is_relevant(case: EvalCase, chunk: StoredChunk) -> bool:
    """Um chunk é relevante se atende a TODOS os critérios informados no caso."""
    if case.fonte and chunk.source != case.fonte and not chunk.source.endswith("/" + case.fonte):
        return False
    if case.pagina is not None and chunk.page != case.pagina:
        return False
    if case.trecho and normalize(case.trecho) not in normalize(chunk.text):
        return False
    return True


def unreachable_cases(cases: Sequence[EvalCase], chunks: Sequence[StoredChunk]) -> list[EvalCase]:
    """Casos que nenhum chunk do índice satisfaz: gabarito errado ou trecho partido entre chunks."""
    return [case for case in cases if not any(is_relevant(case, chunk) for chunk in chunks)]


def evaluate(retriever: Retriever, cases: Sequence[EvalCase], k: int, mode: SearchMode) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        hits = retriever.search(case.pergunta, k=k, mode=mode)
        rank = next((i for i, hit in enumerate(hits, start=1) if is_relevant(case, hit.chunk)), None)
        results.append(CaseResult(case, rank, hits))
    return EvalReport(mode, k, results)
