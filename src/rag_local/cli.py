"""CLI: rag ingest | ask | stats | eval."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from rag_local.config import Settings, load_settings
from rag_local.embeddings import OllamaEmbedder
from rag_local.evaluation import evaluate, load_cases, unreachable_cases
from rag_local.generation import Generator
from rag_local.ingest import ingest as run_ingest
from rag_local.ingest import signature_mismatch
from rag_local.ollama import OllamaError
from rag_local.retrieval import Retriever, SearchHit
from rag_local.store import Store

app = typer.Typer(help="RAG local com Ollama + DuckDB.", no_args_is_help=True, add_completion=False)
console = Console()


class Mode(str, Enum):
    hybrid = "hybrid"
    vector = "vector"
    bm25 = "bm25"


@contextmanager
def _session() -> Iterator[tuple[Settings, Store]]:
    """Carrega a configuração, abre o banco e transforma erros conhecidos em mensagens limpas."""
    try:
        settings = load_settings()
    except ValueError as exc:
        console.print(f"[red]Configuração inválida:[/] {exc}")
        raise typer.Exit(1) from exc
    try:
        with Store(settings.db_path, settings.embed_dim) as store:
            yield settings, store
    except OllamaError as exc:
        console.print(f"[red]Erro no Ollama:[/] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def ingest() -> None:
    """Indexa a pasta de documentos (só reprocessa o que mudou)."""
    with _session() as (settings, store):
        console.print(f"Pasta: [bold]{settings.docs_dir}[/]")
        report = run_ingest(
            settings,
            store,
            OllamaEmbedder.from_settings(settings),
            on_file=lambda source: console.print(f"  indexando [cyan]{source}[/]"),
        )

    if report.created_samples:
        console.print("[yellow]Pasta vazia ou inexistente: criei arquivos de exemplo.[/]")
    if report.full_reindex_reason:
        console.print(f"[yellow]Reindexação completa[/] ({report.full_reindex_reason})")
    for source in report.removed:
        console.print(f"  removido [magenta]{source}[/]")
    for source in report.empty:
        console.print(f"  [yellow]aviso:[/] {source} não tem texto extraível (PDF escaneado?)")
    for source, error in report.failed:
        console.print(f"  [red]falhou[/] {source}: {error}")
    console.print(
        f"[green]Pronto.[/] novos: {len(report.added)}, atualizados: {len(report.updated)}, "
        f"removidos: {len(report.removed)}, sem mudança: {len(report.unchanged)}, "
        f"falhas: {len(report.failed)}, chunks gravados: {report.chunks_written}"
    )


def _require_index(settings: Settings, store: Store) -> None:
    if store.count_chunks() == 0:
        console.print("[yellow]O índice está vazio.[/] Rode [bold]rag ingest[/] primeiro.")
        raise typer.Exit(1)
    mismatch = signature_mismatch(store.get_meta(), settings)
    if mismatch:
        console.print(
            "[yellow]A configuração mudou desde a última ingestão[/] "
            f"({'; '.join(mismatch)}). Rode [bold]rag ingest[/] para reindexar."
        )
        raise typer.Exit(1)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Pergunta em linguagem natural.")],
    k: Annotated[int | None, typer.Option("--k", "-k", help="Quantos chunks usar no contexto.")] = None,
    mode: Annotated[Mode | None, typer.Option("--mode", "-m", help="Tipo de busca.")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Mostra o texto dos chunks e o rank de cada busca.")
    ] = False,
) -> None:
    """Responde uma pergunta com base nos documentos, citando as fontes."""
    with _session() as (settings, store):
        _require_index(settings, store)
        retriever = Retriever.from_settings(store, OllamaEmbedder.from_settings(settings), settings)
        search_mode = (mode or Mode(settings.search_mode)).value
        hits = retriever.search(question, k=k or settings.top_k, mode=search_mode)

        console.print(f"[dim]modelo: {settings.gen_model} | busca: {search_mode}[/]\n")
        for piece in Generator.from_settings(settings).stream_answer(question, hits):
            # Escrita direta (sem rich) para não interpretar [fonte] como markup.
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n\n")

    _print_hits(hits, search_mode, verbose)


def _fmt(value: float | int | None, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def _print_hits(hits: list[SearchHit], mode: str, verbose: bool) -> None:
    table = Table(title="Trechos usados", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Fonte")
    table.add_column("Score", justify="right")
    if verbose and mode == "hybrid":
        table.add_column("Vetor (rank / cos)", justify="right")
        table.add_column("BM25 (rank / score)", justify="right")
    for position, hit in enumerate(hits, start=1):
        row = [str(position), escape(hit.citation), f"{hit.score:.4f}"]
        if verbose and mode == "hybrid":
            row.append(f"{_fmt(hit.vector_rank)} / {_fmt(hit.vector_score, '.3f')}")
            row.append(f"{_fmt(hit.bm25_rank)} / {_fmt(hit.bm25_score, '.2f')}")
        table.add_row(*row)
    console.print(table)

    if verbose:
        for position, hit in enumerate(hits, start=1):
            console.print(
                Panel(
                    escape(hit.chunk.text),
                    title=f"#{position} {escape(hit.citation)} (chunk {hit.chunk.chunk_index})",
                    title_align="left",
                )
            )


@app.command("eval")
def eval_command(
    k: Annotated[int | None, typer.Option("--k", "-k", help="Top k avaliado.")] = None,
    mode: Annotated[
        Mode | None, typer.Option("--mode", "-m", help="Avalia só um modo (padrão: compara os três).")
    ] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Arquivo YAML de perguntas.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Mostra o resultado por pergunta.")] = False,
) -> None:
    """Mede hit rate e MRR do retrieval com eval/perguntas.yaml."""
    with _session() as (settings, store):
        _require_index(settings, store)
        path = file or settings.eval_file
        try:
            cases = load_cases(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            console.print(f"[red]Não consegui ler {path}:[/] {exc}")
            raise typer.Exit(1) from exc
        if not cases:
            console.print(f"[yellow]Nenhuma pergunta em {path}.[/]")
            raise typer.Exit(1)

        for case in unreachable_cases(cases, store.all_chunks()):
            console.print(
                f"[yellow]aviso:[/] nenhum chunk satisfaz o gabarito de {escape(repr(case.pergunta))} "
                "(fonte/página/trecho errados, ou trecho partido entre dois chunks)"
            )

        top_k = k or settings.top_k
        modes = [mode] if mode else list(Mode)
        retriever = Retriever.from_settings(store, OllamaEmbedder.from_settings(settings), settings)
        reports = [evaluate(retriever, cases, top_k, m.value) for m in modes]

    console.print(
        f"[dim]{len(cases)} perguntas | k={top_k} | embedding {settings.embed_model} | "
        f"chunk {settings.chunk_size}/{settings.chunk_overlap} | candidatos {settings.candidates}[/]"
    )
    summary = Table(title="Retrieval")
    summary.add_column("Modo")
    summary.add_column(f"Hit rate@{top_k}", justify="right")
    summary.add_column(f"MRR@{top_k}", justify="right")
    for report in reports:
        summary.add_row(report.mode, f"{report.hit_rate:.1%}", f"{report.mrr:.3f}")
    console.print(summary)

    if verbose:
        detail = Table(title="Rank do primeiro chunk relevante por pergunta (- = não veio no top k)")
        detail.add_column("Pergunta")
        for report in reports:
            detail.add_column(report.mode, justify="right")
        for i, case in enumerate(cases):
            ranks = [report.results[i].rank for report in reports]
            detail.add_row(escape(case.pergunta), *(_fmt(r) for r in ranks))
        console.print(detail)


@app.command()
def stats() -> None:
    """Mostra quantidade de arquivos e chunks indexados."""
    with _session() as (settings, store):
        files = store.list_files()
        total_chunks = store.count_chunks()
        meta = store.get_meta()

    table = Table(title=f"Índice em {settings.db_path.name}")
    table.add_column("Arquivo")
    table.add_column("Chunks", justify="right")
    table.add_column("Tamanho", justify="right")
    table.add_column("Indexado em")
    for info in files:
        table.add_row(
            info.source,
            str(info.n_chunks),
            f"{info.size_bytes / 1024:.1f} KB",
            info.ingested_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)
    console.print(f"Arquivos: [bold]{len(files)}[/]  Chunks: [bold]{total_chunks}[/]")
    if meta:
        console.print(
            f"Embedding: {meta.get('embed_model')} ({meta.get('embed_dim')} dim), "
            f"chunk {meta.get('chunk_size')}/{meta.get('chunk_overlap')}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
