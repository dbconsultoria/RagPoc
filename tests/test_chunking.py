import pytest

from rag_local.chunking import chunk_sections, split_text
from rag_local.loaders import Section, clean_text

LOREM = (
    "O edital define as regras do concurso. As inscrições ficam abertas por trinta dias. "
    "A prova objetiva tem sessenta questões. O resultado sai no site oficial. "
)


def test_short_text_is_single_chunk() -> None:
    assert split_text("Texto curto.", chunk_size=800, overlap=100) == ["Texto curto."]


def test_empty_text_has_no_chunks() -> None:
    assert split_text("   \n\n  ", chunk_size=800, overlap=100) == []


def test_chunks_respect_max_size() -> None:
    chunks = split_text(LOREM * 40, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_prefers_paragraph_break() -> None:
    first = "a" * 70 + " fim do primeiro parágrafo."
    text = first + "\n\n" + "segundo parágrafo " * 10
    chunks = split_text(text, chunk_size=120, overlap=0)
    assert chunks[0] == first


def test_prefers_sentence_end_over_word_break() -> None:
    chunks = split_text(LOREM * 5, chunk_size=150, overlap=0)
    assert all(c.endswith(".") for c in chunks[:-1])


def test_overlap_repeats_text_between_chunks() -> None:
    chunks = split_text(LOREM * 10, chunk_size=200, overlap=60)
    for previous, current in zip(chunks, chunks[1:]):
        head = current[:20]
        assert head in previous, "o início do chunk seguinte deveria estar no anterior"


def test_chunks_do_not_start_mid_word() -> None:
    words = {w.strip(".") for w in LOREM.split()}
    chunks = split_text(LOREM * 10, chunk_size=180, overlap=50)
    for chunk in chunks:
        assert chunk.split()[0].strip(".") in words


def test_text_without_spaces_is_hard_cut() -> None:
    chunks = split_text("x" * 1000, chunk_size=300, overlap=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=size, overlap=overlap)


def test_chunk_sections_keeps_page_and_global_index() -> None:
    sections = [Section(LOREM * 3, page=1), Section(LOREM * 3, page=2)]
    chunks = chunk_sections(sections, chunk_size=150, overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    pages = [c.page for c in chunks]
    assert pages == sorted(pages), "chunks de uma página não podem se misturar com os de outra"


def test_clean_text_joins_hyphenation_and_collapses_blank_lines() -> None:
    raw = "inscri-\nção   aberta\r\n\r\n\r\n\r\nPróximo   parágrafo"
    assert clean_text(raw) == "inscrição aberta\n\nPróximo parágrafo"
