"""Blind questions about the test PDFs, written from the source text before any search was run.
Usage: python write_pdf_questions.py <site-packages> <out-dir>"""

import json
import re
import sys
from pathlib import Path

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def desc(prefix):
    return (next(SP.glob(f"{prefix}-*.dist-info")) / "METADATA").read_text(encoding="utf-8").partition("\n\n")[2]


precise = [
    ("Which Python version does httpcore require?", "Python 3.8+"),
    ("What are httpcore's required dependencies?", "has the following dependencies"),
    ("Why would someone use httpcore directly instead of a higher-level client?", "proxy service in Python"),
    ("What changed in httpcore 1.0.5?", "1.0.5 (March 27th, 2024)"),
    ("Which package is needed for HTTP/2 support in httpcore?", "’h2’ - Required by"),  # PDF turns ' into ’
    ("What is the minimum Python version for Rich?", "Rich requires Python 3.8 or later"),
    ("How do I test Rich output in my terminal?", "test Rich output on your terminal"),
    ("What library does Rich use for syntax highlighting?", "library to implement syntax highlighting"),
    ("Does Rich work in Jupyter notebooks?", "with no additional configuration required"),
    ("What does Rich's inspect function do?", "produce a report on any Python object"),
    ("How many colors does the classic Windows terminal support?", "limited to 16 colors"),
    ("How do I install pandas with conda?", "conda install -c conda-forge pandas"),
    ("Where can I find the list of changes between pandas releases?", "list of changes to pandas between each release"),
    ("What does pandas aim to become?", "most powerful and flexible open-source data"),
    ("How do I auto-format the fsspec code?", "black fsspec"),
    ("What is fsspec?", "A specification for pythonic filesystems"),
    ("How do I set up pre-commit hooks for fsspec?", "pre-commit install --install-hooks"),
]
with (OUT / "q_pdf_precise.jsonl").open("w", encoding="utf-8") as f:
    for q, t in precise:
        f.write(json.dumps({"question": q, "expected_text": t}) + "\n")

rich_sections = re.findall(r"(?m)^#{1,2} (.+)$", desc("rich"))
changelog = re.split(r"(?m)^#+ Changelog", desc("httpcore"), maxsplit=1)[1]
releases = [h.split(" (")[0].replace("Version ", "") for h in re.findall(r"(?m)^## (.+)$", changelog)]
lists = [
    ("Summarize each section of the Rich README.", rich_sections),
    ("List every httpcore release in the changelog.", [f"{r} (" for r in releases]),
]
with (OUT / "q_pdf_lists.jsonl").open("w", encoding="utf-8") as f:
    for q, items in lists:
        f.write(json.dumps({"question": q, "expected_items": items}) + "\n")
print(f"{len(precise)} precise, lists: {[len(i) for _, i in lists]} items")
