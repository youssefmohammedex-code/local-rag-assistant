"""Third blind list set, written BEFORE running the full-path summaries change. Never used for design."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from rag.loaders import read_text_file, split_markdown  # noqa: E402

OUT = Path(sys.argv[1])
DOCS = OUT / "docs"


def children(file, parent_last, startswith=False):
    kids = []
    for path, _ in split_markdown(read_text_file(DOCS / file)):
        parts = path.split(" > ")
        for i, part in enumerate(parts[:-1]):
            if part == parent_last or (startswith and part.startswith(parent_last)):
                if parts[i + 1] not in kids:
                    kids.append(parts[i + 1])
                break
    return kids


questions = [
    ("List all the option flags orjson supports.", children("orjson.md", "option")),
    ("What are all the ways to install charset-normalizer?", children("charset-normalizer.md", "✨ Installation")),
    ("Walk me through each part of the Typer example upgrade.", children("typer.md", "Example upgrade")),
    ("Summarize every part of the pydantic v2.14.0 release notes.", children("pydantic.md", "v2.14.0 (", True)),
    ("List all the subsections of the Typer installation instructions.", children("typer.md", "Installation")),
]
with (OUT / "q_lists_fresh.jsonl").open("w", encoding="utf-8") as f:
    for q, items in questions:
        f.write(json.dumps({"question": q, "expected_items": items}, ensure_ascii=False) + "\n")
        print(f"{q}\n   {len(items)} items: {items}")
