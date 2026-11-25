import pytest

from rag.chunking import chunk_pages, normalize, split_text
from rag.loaders import Page

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))


def test_chunks_respect_max_size():
    chunks = split_text(TEXT, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_every_word_is_covered():
    chunks = split_text(TEXT, chunk_size=250, overlap=40)
    joined = " ".join(chunks)
    for i in range(200):
        assert f"Sentence number {i} " in joined


def test_consecutive_chunks_overlap():
    chunks = split_text(TEXT, chunk_size=300, overlap=80)
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[0] in a  # the next chunk starts with text from the previous one


def test_zero_overlap_has_no_duplication():
    chunks = split_text(TEXT, chunk_size=300, overlap=0)
    assert sum(len(c) for c in chunks) <= len(TEXT)


def test_prefers_sentence_boundaries():
    chunks = split_text(TEXT, chunk_size=300, overlap=0)
    assert all(c.endswith(".") for c in chunks)


def test_short_and_empty_text():
    assert split_text("hello world", 100, 10) == ["hello world"]
    assert split_text("   \n\n  ", 100, 10) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_arguments(size, overlap):
    with pytest.raises(ValueError):
        split_text(TEXT, size, overlap)


def test_normalize_collapses_whitespace():
    assert normalize("a  \t b\r\n\n\n\nc") == "a b\n\nc"


def test_chunk_ids_are_unique_and_keep_metadata():
    pages = [Page("a.md", 1, TEXT), Page("b.pdf", 3, TEXT)]
    chunks = chunk_pages(pages, 400, 60)
    assert len({c.id for c in chunks}) == len(chunks)
    assert {(c.source, c.page) for c in chunks} == {("a.md", 1), ("b.pdf", 3)}
