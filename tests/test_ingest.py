"""Indexing safety: a failed or empty re-index must never destroy a working index."""

import pytest
from conftest import make_pdf

from rag.errors import NoDocumentsError, friendly_error
from rag.loaders import load_documents


def write_docs(pipeline, **files):
    for name, text in files.items():
        (pipeline.settings.docs_dir / name).write_text(text, encoding="utf-8")


def test_ingest_reports_chunks_and_files(pipeline):
    write_docs(pipeline, **{"a.md": "# A\n\nAlpha text.", "b.txt": "Bravo text."})
    report = pipeline.ingest()
    assert (report.chunks, report.files, report.skipped) == (2, 2, [])
    assert pipeline.store.count() == 2
    assert pipeline.store.sources() == ["a.md", "b.txt"]


def test_failed_embedding_keeps_existing_index(pipeline):
    write_docs(pipeline, **{"a.md": "# A\n\nAlpha text.", "b.txt": "Bravo text."})
    pipeline.ingest()

    def ollama_down(texts, kind="document"):
        raise ConnectionError("Failed to connect to Ollama.")

    pipeline.store.embed = ollama_down
    write_docs(pipeline, **{"c.txt": "Charlie text."})
    with pytest.raises(ConnectionError):
        pipeline.ingest()
    assert pipeline.store.count() == 2  # old index untouched
    assert pipeline.store.sources() == ["a.md", "b.txt"]


def test_empty_library_raises_and_keeps_index(pipeline):
    write_docs(pipeline, **{"a.md": "# A\n\nAlpha text."})
    pipeline.ingest()
    (pipeline.settings.docs_dir / "a.md").unlink()
    with pytest.raises(NoDocumentsError, match="left unchanged"):
        pipeline.ingest()
    assert pipeline.store.count() == 1


def test_reingest_replaces_rather_than_duplicates(pipeline):
    write_docs(pipeline, **{"a.txt": "Alpha text."})
    pipeline.ingest()
    pipeline.ingest()
    assert pipeline.store.count() == 1


def test_search_returns_hits_with_metadata(pipeline):
    write_docs(pipeline, **{"guide.md": "# Guide\n\n## Rules\nRule one."})
    pipeline.ingest()
    hit = pipeline.retrieve("Rule one", k=3)[0]
    assert (hit.source, hit.section) == ("guide.md", "Guide > Rules")
    assert hit.text.startswith("Guide > Rules\n\n")


def test_unreadable_files_are_reported_not_silently_dropped(tmp_path):
    make_pdf(tmp_path / "text.pdf", ["Parking permits cost 300 EGP."])
    make_pdf(tmp_path / "scanned.pdf", [])
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "headings-only.md").write_text("# Title\n## Section", encoding="utf-8")
    skipped: list[str] = []
    pages = load_documents(tmp_path, skipped=skipped)
    assert [(p.source, p.page) for p in pages] == [("text.pdf", 1)]
    assert "300 EGP" in pages[0].text
    assert sorted(skipped) == ["broken.pdf", "empty.txt", "headings-only.md", "scanned.pdf"]


def test_windows_1252_text_keeps_its_characters(tmp_path):
    (tmp_path / "legacy.txt").write_bytes("Café “quotes” – €5".encode("cp1252"))
    assert load_documents(tmp_path)[0].text == "Café “quotes” – €5"


def test_utf8_bom_is_removed(tmp_path):
    (tmp_path / "bom.txt").write_bytes("﻿Hello".encode())
    assert load_documents(tmp_path)[0].text == "Hello"


def test_unsupported_and_hidden_files(tmp_path):
    for name in ["report.docx", "page.html", "~$lock.md", ".hidden.md", ".obsidian/workspace.md", "ok.md"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("# T\n\nText.", encoding="utf-8")
    unsupported: list[str] = []
    pages = load_documents(tmp_path, unsupported=unsupported)
    assert [p.source for p in pages] == ["ok.md"]  # lock/hidden files ignored entirely
    assert sorted(unsupported) == ["page.html", "report.docx"]


def test_ingest_report_lists_unsupported_files(pipeline):
    write_docs(pipeline, **{"a.txt": "Alpha text.", "b.docx": "fake"})
    assert pipeline.ingest().unsupported == ["b.docx"]


def test_ingest_report_lists_skipped_files(pipeline):
    write_docs(pipeline, **{"a.txt": "Alpha text."})
    make_pdf(pipeline.settings.docs_dir / "scan.pdf", [])
    assert pipeline.ingest().skipped == ["scan.pdf"]


@pytest.mark.parametrize("exc,expected", [
    (ConnectionError("Failed to connect to Ollama"), "Can't reach Ollama"),
    (NoDocumentsError("No readable documents found"), "No readable documents found"),
    (FileNotFoundError("Docs folder not found: x"), "Docs folder not found"),
    (RuntimeError("boom"), "Unexpected error (RuntimeError): boom"),
])
def test_friendly_error_messages(exc, expected):
    assert expected in friendly_error(exc)


@pytest.mark.parametrize("message", [
    "model 'qwen2.5:14b' not found",  # what Ollama 0.40 actually returns
    'model "qwen2.5:14b" not found, try pulling it first',
])
def test_friendly_error_for_missing_model(message):
    import ollama
    expected = "The model `qwen2.5:14b` isn't installed. Run `ollama pull qwen2.5:14b` and try again."
    assert friendly_error(ollama.ResponseError(message, 404)) == expected
