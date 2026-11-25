"""Second blind list set, written BEFORE changing the list-expansion scoring (held out from tuning)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from rag.loaders import read_text_file, split_markdown  # noqa: E402

OUT = Path(sys.argv[1])
DOCS = OUT / "docs"


def sections(file):
    return [p for p, _ in split_markdown(read_text_file(DOCS / file))]


def children(file, parent_last):
    kids = []
    for path in sections(file):
        parts = path.split(" > ")
        if parent_last in parts[:-1]:
            kid = parts[parts.index(parent_last) + 1]
            if kid not in kids:
                kids.append(kid)
    return kids


typer_top = []
for p in sections("typer.md"):
    top = p.split(" > ")[0]
    if top and top not in typer_top:
        typer_top.append(top)
charset_top = []
for p in sections("charset-normalizer.md"):
    top = p.split(" > ")[0]
    if top and top not in charset_top and top != "Changelog":
        charset_top.append(top)
pydantic_214 = [p.split(" > ")[1].split(" (")[0] for p in sections("pydantic.md")
                if p.count(" > ") >= 1 and p.split(" > ")[1].startswith("v2.14.")]
pydantic_214 = list(dict.fromkeys(pydantic_214))

usage_parent = next(p.split(" > ")[-2] for p in sections("orjson.md") if p.endswith(" > Quickstart"))
questions = [
    ("List all the sections of the Typer README.", typer_top),
    ("List all the other use cases python-dotenv describes.", children("python-dotenv.md", "Other Use Cases")),
    ("Describe each top-level section of the charset-normalizer documentation.", charset_top),
    ("List every 2.14 release of pydantic.", pydantic_214),
    (f"What does the orjson {usage_parent} section cover? List all its parts.", children("orjson.md", usage_parent)),
]
with (OUT / "q_lists_heldout.jsonl").open("w", encoding="utf-8") as f:
    for q, items in questions:
        f.write(json.dumps({"question": q, "expected_items": items}, ensure_ascii=False) + "\n")
        print(f"{q}\n   {len(items)} items: {items}")
