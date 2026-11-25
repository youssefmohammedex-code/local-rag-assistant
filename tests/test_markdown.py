from rag.chunking import chunk_pages
from rag.loaders import load_documents, pretty_source, split_markdown

DOC = """# Project Guide

Intro paragraph.

# 1. Modules

## Module A — Engine (Alice)
Owns the tick loop.

## Module B — World (Bob)
Owns the map.

```python
# not a heading
x = 1
```

# 2. Testing
Run pytest.
"""


def test_sections_get_full_heading_paths():
    sections = dict(split_markdown(DOC))
    assert sections["Project Guide"] == "Intro paragraph."
    assert sections["Project Guide > 1. Modules > Module A — Engine (Alice)"] == "Owns the tick loop."
    assert sections["Project Guide > 2. Testing"] == "Run pytest."


def test_headings_inside_code_blocks_are_ignored():
    sections = dict(split_markdown(DOC))
    body = sections["Project Guide > 1. Modules > Module B — World (Bob)"]
    assert "# not a heading" in body


def test_heading_only_sections_are_skipped():
    paths = [p for p, _ in split_markdown(DOC)]
    assert "Project Guide > 1. Modules" not in paths


def test_markdown_chunks_carry_contextual_header(tmp_path):
    (tmp_path / "guide.md").write_text(DOC, encoding="utf-8")
    chunks = chunk_pages(load_documents(tmp_path), 500, 50)
    engine = next(c for c in chunks if c.section.endswith("Module A — Engine (Alice)"))
    assert engine.text.startswith("Project Guide > 1. Modules > Module A — Engine (Alice)\n\n")
    assert engine.section.endswith("Module A — Engine (Alice)")
    assert len({c.id for c in chunks}) == len(chunks)


def test_setext_headings_are_recognized():
    text = "Title\n=====\n\nIntro.\n\nPart One\n--------\nAlpha text.\n"
    assert dict(split_markdown(text)) == {"Title": "Intro.", "Title > Part One": "Alpha text."}


def test_horizontal_rule_after_a_paragraph_is_not_a_heading():
    text = "# Doc\n\nLine one of a paragraph\nline two\n---\nAfter the rule.\n"
    sections = dict(split_markdown(text))
    assert list(sections) == ["Doc"] and "After the rule." in sections["Doc"]


def test_table_separator_is_not_a_setext_heading():
    text = "# Doc\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    assert list(dict(split_markdown(text))) == ["Doc"]


def test_front_matter_is_removed():
    text = "---\ntitle: Notes\ntags: [a]\n---\n# Notes\n\nBody.\n"
    assert split_markdown(text) == [("Notes", "Body.")]


def test_bom_does_not_hide_the_first_heading():
    assert split_markdown("﻿# Title\n\nBody.") == [("Title", "Body.")]


def test_pretty_source_strips_notion_ids_and_folders():
    assert pretty_source("notes/Architecture v2 421ffbd20ecf8156af6af5e02dca24c2.md") == "Architecture v2"
    assert pretty_source("sample/student-handbook.md") == "student-handbook"
