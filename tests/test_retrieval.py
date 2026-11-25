from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from rag_local.config import Settings
from rag_local.embeddings import OllamaEmbedder
from rag_local.generation import NOT_FOUND, Generator
from rag_local.ingest import ingest
from rag_local.ollama import OllamaClient, OllamaError
from rag_local.retrieval import Retriever, reciprocal_rank_fusion
from rag_local.store import Store
from conftest import FakeEmbedder


@pytest.fixture
def indexed(settings: Settings, store: Store, embedder: FakeEmbedder) -> Retriever:
    report = ingest(settings, store, embedder)
    assert report.created_samples
    return Retriever(store, embedder, candidates=10)


# ------------------------------------------------------------------- RRF


def test_rrf_sums_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert [doc_id for doc_id, _ in fused] == [1, 3, 2]


def test_rrf_with_empty_ranking_keeps_other_order() -> None:
    assert [d for d, _ in reciprocal_rank_fusion([[5, 4], []])] == [5, 4]


# --------------------------------------------------------------- ingestão


def test_ingest_is_incremental(settings: Settings, store: Store, embedder: FakeEmbedder) -> None:
    first = ingest(settings, store, embedder)
    assert len(first.added) == 3 and first.chunks_written > 0
    embedded_after_first = embedder.documents_embedded

    second = ingest(settings, store, embedder)
    assert not second.changed and len(second.unchanged) == 3
    assert embedder.documents_embedded == embedded_after_first, "nada deveria ser reembedado"

    (settings.docs_dir / "exemplo-duckdb.txt").write_text("DuckDB mudou de texto.", encoding="utf-8")
    (settings.docs_dir / "exemplo-ollama.md").unlink()
    third = ingest(settings, store, embedder)
    assert third.updated == ["exemplo-duckdb.txt"]
    assert third.removed == ["exemplo-ollama.md"]
    assert {f.source for f in store.list_files()} == {"exemplo-duckdb.txt", "exemplo-rag.md"}
    assert all(c.source != "exemplo-ollama.md" for c in store.all_chunks())


def test_subfolders_use_relative_posix_source(settings: Settings, store: Store, embedder: FakeEmbedder) -> None:
    nested = settings.docs_dir / "sub" / "pasta"
    nested.mkdir(parents=True)
    (nested / "nota.txt").write_text("Uma nota dentro de subpasta.", encoding="utf-8")
    ingest(settings, store, embedder)
    assert "sub/pasta/nota.txt" in {f.source for f in store.list_files()}


def test_changing_chunk_size_forces_full_reindex(
    settings: Settings, store: Store, embedder: FakeEmbedder
) -> None:
    ingest(settings, store, embedder)
    report = ingest(replace(settings, chunk_size=200), store, embedder)
    assert report.full_reindex_reason and "chunk_size" in report.full_reindex_reason
    assert len(report.added) == 3


def test_bm25_index_follows_deletions(settings: Settings, store: Store, embedder: FakeEmbedder) -> None:
    ingest(settings, store, embedder)
    assert store.bm25_search("Ollama", 10)
    for path in settings.docs_dir.iterdir():
        path.unlink()
    ingest(settings, store, embedder)
    assert store.count_chunks() == 0
    assert store.bm25_search("Ollama", 10) == []


# ------------------------------------------------------------------ busca


def test_vector_search_finds_relevant_document(indexed: Retriever) -> None:
    hits = indexed.search("array_cosine_similarity FLOAT vetores DuckDB", k=2, mode="vector")
    assert hits[0].chunk.source == "exemplo-duckdb.txt"
    assert hits[0].score > hits[-1].score or len(hits) == 1


def test_bm25_ignores_accents(indexed: Retriever) -> None:
    # Texto tem "dimensões"; a pergunta vem sem acento.
    hits = indexed.search("dimensoes", k=1, mode="bm25")
    assert hits and hits[0].chunk.source == "exemplo-ollama.md"


def test_bm25_uses_portuguese_stemming(indexed: Retriever) -> None:
    # Texto tem "recuperados"; a pergunta usa o singular.
    hits = indexed.search("recuperado", k=1, mode="bm25")
    assert hits and hits[0].chunk.source == "exemplo-rag.md"


def test_bm25_ignores_stopwords(indexed: Retriever) -> None:
    assert indexed.search("o de que para", k=3, mode="bm25") == []


def test_hybrid_reports_ranks_from_both_searches(indexed: Retriever) -> None:
    hits = indexed.search("Reciprocal Rank Fusion", k=3, mode="hybrid")
    assert hits[0].chunk.source == "exemplo-rag.md"
    assert hits[0].vector_rank is not None and hits[0].bm25_rank is not None
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_respects_k(indexed: Retriever) -> None:
    assert len(indexed.search("documentos", k=2)) == 2


def test_citation_includes_page_only_for_pdf(indexed: Retriever) -> None:
    hit = indexed.search("DuckDB", k=1, mode="bm25")[0]
    assert hit.citation == hit.chunk.source  # .txt não tem página


# ------------------------------------------------------- Ollama mockado


def _mock_client(handler) -> OllamaClient:
    return OllamaClient("http://ollama.test", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ollama_embedder_batches_and_applies_prefixes() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1] * 768 for _ in body["input"]]})

    embedder = OllamaEmbedder(
        _mock_client(handler), "nomic-embed-text", 768, "search_document: ", "search_query: ", batch_size=2
    )
    assert len(embedder.embed_documents(["a", "b", "c"])) == 3
    embedder.embed_query("pergunta")

    assert [len(r["input"]) for r in requests] == [2, 1, 1]
    assert requests[0]["input"][0] == "search_document: a"
    assert requests[-1]["input"] == ["search_query: pergunta"]


def test_ollama_embedder_rejects_wrong_dimension() -> None:
    client = _mock_client(lambda r: httpx.Response(200, json={"embeddings": [[0.1] * 384]}))
    with pytest.raises(OllamaError, match="384"):
        OllamaEmbedder(client, "outro-modelo", 768).embed_query("x")


def test_missing_model_gives_pull_hint() -> None:
    client = _mock_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        OllamaEmbedder(client, "nomic-embed-text", 768).embed_query("x")


def test_generator_streams_content_and_sends_context(indexed: Retriever) -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        lines = [
            {"message": {"thinking": "pensando..."}},
            {"message": {"content": "Resposta "}},
            {"message": {"content": "[exemplo-rag.md]"}, "done": True},
        ]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    hits = indexed.search("Reciprocal Rank Fusion", k=2)
    answer = "".join(Generator(_mock_client(handler), "modelo").stream_answer("O que é RRF?", hits))
    assert answer == "Resposta [exemplo-rag.md]"
    user_message = sent[0]["messages"][1]["content"]
    assert "[exemplo-rag.md]" in user_message and "O que é RRF?" in user_message


def test_generator_without_hits_does_not_call_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("não deveria chamar o Ollama")

    assert "".join(Generator(_mock_client(handler), "m").stream_answer("?", [])) == NOT_FOUND


# -------------------------------------------------------------- avaliação


def test_eval_metrics_and_relevance(indexed: Retriever, tmp_path) -> None:
    from rag_local.evaluation import evaluate, load_cases, unreachable_cases

    path = tmp_path / "perguntas.yaml"
    path.write_text(
        """
perguntas:
  - pergunta: O que é Reciprocal Rank Fusion?
    fonte: exemplo-rag.md
    trecho: "Reciprocal Rank   Fusion"
  - pergunta: Como criar o índice textual?
    fonte: exemplo-duckdb.txt
    trecho: create_fts_index
  - pergunta: Pergunta com gabarito impossível
    fonte: nao-existe.md
""",
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [c.pergunta for c in unreachable_cases(cases, indexed.store.all_chunks())] == [
        "Pergunta com gabarito impossível"
    ]

    report = evaluate(indexed, cases, k=3, mode="bm25")
    ranks = [r.rank for r in report.results]
    assert ranks[0] is not None and ranks[1] is not None and ranks[2] is None
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 / ranks[0] + 1 / ranks[1]) / 3)


def test_eval_rejects_case_without_gold(tmp_path) -> None:
    from rag_local.evaluation import load_cases

    path = tmp_path / "ruim.yaml"
    path.write_text("perguntas:\n  - pergunta: sem gabarito\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fonte"):
        load_cases(path)
