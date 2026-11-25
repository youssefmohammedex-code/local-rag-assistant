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
    for a, b in zip(chunks, chunks[1:], strict=False):
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


SMALL = [Page("d.md", 1, "Python 3.8+", section="Lib > Requirements", part=1),
         Page("d.md", 1, "pip install lib " * 20, section="Lib > Installation", part=2),
         Page("d.md", 1, "Long usage text. " * 60, section="Lib > Usage", part=3)]


def test_tiny_section_borrows_context_but_keeps_its_own_heading():
    chunks = chunk_pages(SMALL, 1000, 150)
    req = next(c for c in chunks if c.section == "Lib > Requirements")
    assert req.text.startswith("Lib > Requirements\n\nPython 3.8+\n\nInstallation\npip install lib")
    assert [c.section for c in chunks] == ["Lib > Requirements", "Lib > Installation", "Lib > Usage", "Lib > Usage"]


def test_borrowing_stops_at_the_file_boundary():
    pages = [Page("a.md", 1, "tiny", section="A > X", part=1), Page("b.md", 1, "other file", section="B", part=1)]
    assert chunk_pages(pages, 1000, 150)[0].text == "A > X\n\ntiny"


def test_small_section_modes_off_and_pack():
    assert chunk_pages(SMALL, 1000, 150, small="off")[0].text == "Lib > Requirements\n\nPython 3.8+"
    packed = chunk_pages(SMALL, 1000, 150, small="pack")
    assert packed[0].section == "Lib"  # merged sections lose their own heading: why "pack" isn't the default


def test_chunk_ids_are_unique_and_keep_metadata():
    pages = [Page("a.md", 1, TEXT), Page("b.pdf", 3, TEXT)]
    chunks = chunk_pages(pages, 400, 60)
    assert len({c.id for c in chunks}) == len(chunks)
    assert {(c.source, c.page) for c in chunks} == {("a.md", 1), ("b.pdf", 3)}
