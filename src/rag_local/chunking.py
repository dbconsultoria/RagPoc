"""Chunking por caracteres com sobreposição, preferindo quebrar em parágrafo ou frase."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rag_local.loaders import Section

_SENTENCE_END = re.compile(r"[.!?…;:][)\"'»\]]*\s")
_WHITESPACE = re.compile(r"\s")


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None
    chunk_index: int


def _find_break(text: str, lo: int, hi: int) -> int:
    """Melhor ponto de corte em text[lo:hi]: parágrafo > frase > linha > espaço > corte seco."""
    window = text[lo:hi]
    paragraph = window.rfind("\n\n")
    if paragraph != -1:
        return lo + paragraph + 2
    last_sentence = None
    for last_sentence in _SENTENCE_END.finditer(window):
        pass
    if last_sentence is not None:
        return lo + last_sentence.end()
    for separator in ("\n", " "):
        position = window.rfind(separator)
        if position != -1:
            return lo + position + 1
    return hi


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size precisa ser positivo")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap precisa ser >= 0 e menor que chunk_size")

    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Só procura quebra na segunda metade da janela, para não gerar chunks minúsculos.
            end = _find_break(text, start + chunk_size // 2, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break

        next_start = end - overlap
        # Começa o próximo chunk no início de uma palavra, não no meio dela.
        boundary = _WHITESPACE.search(text, next_start, end) if overlap else None
        if boundary is not None:
            next_start = boundary.end()
        start = next_start if next_start > start else end
    return chunks


def chunk_sections(sections: Iterable[Section], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunks nunca atravessam seções, então a página de cada chunk é exata."""
    chunks: list[Chunk] = []
    for section in sections:
        for piece in split_text(section.text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, page=section.page, chunk_index=len(chunks)))
    return chunks
