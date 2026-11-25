"""More blind PDF list questions, written BEFORE trying any fix for PDF list questions.
Usage: python write_pdf_list_heldout.py <site-packages> <out-dir>"""

import json
import re
import sys
from pathlib import Path

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def desc(prefix):
    return (next(SP.glob(f"{prefix}-*.dist-info")) / "METADATA").read_text(encoding="utf-8").partition("\n\n")[2]


def headings(markdown, level):
    out, in_code = [], False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = re.match(rf"^#{{{level}}} (.+)$", line)
        if m and not in_code:
            out.append(m.group(1).strip())
    return out


httpcore = desc("httpcore")
top = [h for h in headings(httpcore, 2) if not re.match(r"^(Version )?\d", h)]  # skip release headings
release_100 = httpcore.split("## 1.0.0 (October 6th, 2023)", 1)[1].split("\n## ", 1)[0]
changes_100 = headings(release_100, 3)
fsspec_sections = headings(desc("fsspec"), 2)

# Dropped after generation, before any search: httpcore 1.0.0 has no sub-headings (empty list), and the
# fsspec section names ("Install", "Support") are common words, so their coverage would be meaningless.
questions = [
    ("What are all the sections of the httpcore guide?", top),
]
del changes_100, fsspec_sections
with (OUT / "q_pdf_lists_heldout.jsonl").open("w", encoding="utf-8") as f:
    for q, items in questions:
        f.write(json.dumps({"question": q, "expected_items": items}) + "\n")
        print(f"{q}\n   {len(items)}: {items}")
