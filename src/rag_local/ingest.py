"""Ingestão incremental: varre a pasta, compara hashes e só reprocessa o que mudou."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag_local.chunking import chunk_sections
from rag_local.config import Settings
from rag_local.embeddings import Embedder
from rag_local.loaders import iter_documents, load_file, relative_source
from rag_local.ollama import OllamaError
from rag_local.samples import SAMPLE_DOCS
from rag_local.store import Store

# Parâmetros que, se mudarem, invalidam todos os chunks/vetores já gravados.
INDEX_PARAMS = ("embed_model", "embed_dim", "embed_doc_prefix", "chunk_size", "chunk_overlap")


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_written: int = 0
    full_reindex_reason: str | None = None
    created_samples: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def index_signature(settings: Settings) -> dict[str, str]:
    return {name: str(getattr(settings, name)) for name in INDEX_PARAMS}


def signature_mismatch(stored: dict[str, str], settings: Settings) -> list[str]:
    """Lista os parâmetros cujo valor gravado no banco difere da configuração atual."""
    current = index_signature(settings)
    return [
        f"{name}: {stored[name]!r} -> {current[name]!r}"
        for name in INDEX_PARAMS
        if name in stored and stored[name] != current[name]
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_docs_dir(docs_dir: Path) -> bool:
    """Cria a pasta com exemplos se ela não existe ou está vazia. Retorna True se criou."""
    if docs_dir.exists() and any(docs_dir.iterdir()):
        return False
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLE_DOCS.items():
        (docs_dir / name).write_text(content, encoding="utf-8")
    return True


def ingest(
    settings: Settings,
    store: Store,
    embedder: Embedder,
    on_file: Callable[[str], None] | None = None,
) -> IngestReport:
    # Exemplos só no primeiro uso: se o índice já teve arquivos, esvaziar a pasta deve
    # limpar o índice, não fazer os exemplos reaparecerem.
    first_run = not store.file_hashes()
    report = IngestReport(created_samples=first_run and ensure_docs_dir(settings.docs_dir))

    mismatch = signature_mismatch(store.get_meta(), settings)
    if mismatch:
        report.full_reindex_reason = "; ".join(mismatch)
        store.reset()

    stored = store.file_hashes()
    current = {relative_source(p, settings.docs_dir): p for p in iter_documents(settings.docs_dir)}

    for source in sorted(stored.keys() - current.keys()):
        store.delete_file(source)
        report.removed.append(source)

    for source, path in current.items():
        digest = file_sha256(path)
        if stored.get(source) == digest:
            report.unchanged.append(source)
            continue
        if on_file:
            on_file(source)
        try:
            chunks = chunk_sections(load_file(path), settings.chunk_size, settings.chunk_overlap)
            vectors = embedder.embed_documents([c.text for c in chunks]) if chunks else []
        except OllamaError:
            raise  # problema global (Ollama fora do ar, modelo ausente): não adianta seguir
        except Exception as exc:  # arquivo corrompido etc.: registra e segue com os outros
            report.failed.append((source, f"{type(exc).__name__}: {exc}"))
            continue

        store.replace_file(source, digest, path.stat().st_size, chunks, vectors)
        (report.updated if source in stored else report.added).append(source)
        if not chunks:
            report.empty.append(source)
        report.chunks_written += len(chunks)

    if report.changed or not store.has_fts():
        store.rebuild_fts()
    store.set_meta(index_signature(settings))
    return report
