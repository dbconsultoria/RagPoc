"""Armazenamento em DuckDB: arquivos, chunks com embedding, metadados do índice e índice FTS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType

import duckdb

from rag_local.chunking import Chunk

FTS_SCHEMA = "fts_main_chunks"

# Stopwords comuns do português (sem acento, porque o FTS remove acentos antes de comparar).
# "nao" fica de fora de propósito: negação muda o sentido da pergunta.
PORTUGUESE_STOPWORDS = """
a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
para pra com sem sob ao aos e ou mas que se como quando onde qual quais quem cujo
este esta estes estas esse essa esses essas isso isto aquele aquela aquilo
eu tu ele ela nos vos eles elas me te lhe lhes seu sua seus suas meu minha
ser e era foi sao sera estar esta estao ter tem tinha ha havia
ja tambem so mais muito muita muitos muitas entre ate apos sobre
""".split()


@dataclass(frozen=True)
class StoredChunk:
    id: int
    source: str
    page: int | None
    chunk_index: int
    text: str


@dataclass(frozen=True)
class FileInfo:
    source: str
    sha256: str
    size_bytes: int
    n_chunks: int
    ingested_at: datetime


class Store:
    def __init__(self, db_path: Path | str, embed_dim: int = 768) -> None:
        self.embed_dim = int(embed_dim)
        self.con = duckdb.connect(str(db_path))
        # INSTALL baixa a extensão só na primeira vez; depois é no-op.
        self.con.execute("INSTALL fts")
        self.con.execute("LOAD fts")
        self._create_schema()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # ---------------------------------------------------------------- schema

    def _create_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                source      VARCHAR PRIMARY KEY,
                sha256      VARCHAR   NOT NULL,
                size_bytes  BIGINT    NOT NULL,
                n_chunks    INTEGER   NOT NULL,
                ingested_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            )
            """
        )
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq")
        # Sem FOREIGN KEY para files: no DuckDB ela dificulta apagar/reinserir;
        # a consistência é garantida por transação em replace_file/delete_file.
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGINT PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                source      VARCHAR NOT NULL,
                page        INTEGER,
                chunk_index INTEGER NOT NULL,
                text        VARCHAR NOT NULL,
                embedding   FLOAT[{self.embed_dim}] NOT NULL
            )
            """
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
        )
        self.con.execute("CREATE TABLE IF NOT EXISTS stopwords_pt (sw VARCHAR)")
        if self.con.execute("SELECT count(*) FROM stopwords_pt").fetchone()[0] == 0:
            self.con.executemany("INSERT INTO stopwords_pt VALUES (?)", [(w,) for w in PORTUGUESE_STOPWORDS])

    def reset(self) -> None:
        """Apaga todo o índice (usado quando modelo de embedding ou chunking mudam)."""
        self.drop_fts()
        for table in ("chunks", "files", "meta"):
            self.con.execute(f"DROP TABLE IF EXISTS {table}")
        self._create_schema()

    # ------------------------------------------------------------------ meta

    def get_meta(self) -> dict[str, str]:
        return dict(self.con.execute("SELECT key, value FROM meta").fetchall())

    def set_meta(self, values: dict[str, str]) -> None:
        self.con.executemany(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            list(values.items()),
        )

    # ----------------------------------------------------------------- files

    def file_hashes(self) -> dict[str, str]:
        return dict(self.con.execute("SELECT source, sha256 FROM files").fetchall())

    def list_files(self) -> list[FileInfo]:
        rows = self.con.execute(
            "SELECT source, sha256, size_bytes, n_chunks, ingested_at FROM files ORDER BY source"
        ).fetchall()
        return [FileInfo(*row) for row in rows]

    def replace_file(
        self,
        source: str,
        sha256: str,
        size_bytes: int,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks e embeddings precisam ter o mesmo tamanho")
        self.con.begin()
        try:
            self.con.execute("DELETE FROM chunks WHERE source = ?", [source])
            if chunks:
                self.con.executemany(
                    f"""
                    INSERT INTO chunks (source, page, chunk_index, text, embedding)
                    VALUES (?, ?, ?, ?, ?::FLOAT[{self.embed_dim}])
                    """,
                    [
                        (source, c.page, c.chunk_index, c.text, list(e))
                        for c, e in zip(chunks, embeddings, strict=True)
                    ],
                )
            self.con.execute(
                """
                INSERT INTO files (source, sha256, size_bytes, n_chunks, ingested_at)
                VALUES (?, ?, ?, ?, current_timestamp)
                ON CONFLICT (source) DO UPDATE SET
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    n_chunks = excluded.n_chunks,
                    ingested_at = excluded.ingested_at
                """,
                [source, sha256, size_bytes, len(chunks)],
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def delete_file(self, source: str) -> None:
        self.con.begin()
        try:
            self.con.execute("DELETE FROM chunks WHERE source = ?", [source])
            self.con.execute("DELETE FROM files WHERE source = ?", [source])
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    # ---------------------------------------------------------------- chunks

    def count_chunks(self) -> int:
        return self.con.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def get_chunks(self, ids: Sequence[int]) -> dict[int, StoredChunk]:
        if not ids:
            return {}
        rows = self.con.execute(
            "SELECT id, source, page, chunk_index, text FROM chunks WHERE list_contains(?, id)",
            [list(ids)],
        ).fetchall()
        return {row[0]: StoredChunk(*row) for row in rows}

    def all_chunks(self) -> list[StoredChunk]:
        rows = self.con.execute(
            "SELECT id, source, page, chunk_index, text FROM chunks ORDER BY source, chunk_index"
        ).fetchall()
        return [StoredChunk(*row) for row in rows]

    # ---------------------------------------------------------------- busca

    def vector_search(self, query_vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        return self.con.execute(
            f"""
            SELECT id, array_cosine_similarity(embedding, ?::FLOAT[{self.embed_dim}]) AS score
            FROM chunks
            ORDER BY score DESC, id
            LIMIT ?
            """,
            [list(query_vector), limit],
        ).fetchall()

    def has_fts(self) -> bool:
        return (
            self.con.execute(
                "SELECT count(*) FROM duckdb_schemas() WHERE schema_name = ?", [FTS_SCHEMA]
            ).fetchone()[0]
            > 0
        )

    def rebuild_fts(self) -> None:
        """O índice FTS do DuckDB não se atualiza sozinho: é recriado após cada ingestão."""
        if self.count_chunks() == 0:
            self.drop_fts()
            return
        # ignore: o padrão do DuckDB descarta dígitos; aqui eles ficam (datas, itens, valores).
        self.con.execute(
            r"""
            PRAGMA create_fts_index(
                'chunks', 'id', 'text',
                stemmer = 'portuguese',
                stopwords = 'stopwords_pt',
                ignore = '(\.|[^a-z0-9])+',
                strip_accents = 1,
                lower = 1,
                overwrite = 1
            )
            """
        )

    def drop_fts(self) -> None:
        if self.has_fts():
            self.con.execute("PRAGMA drop_fts_index('chunks')")

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        if not query.strip() or not self.has_fts():
            return []
        return self.con.execute(
            f"""
            SELECT id, score FROM (
                SELECT id, {FTS_SCHEMA}.match_bm25(id, ?) AS score FROM chunks
            )
            WHERE score IS NOT NULL
            ORDER BY score DESC, id
            LIMIT ?
            """,
            [query, limit],
        ).fetchall()
