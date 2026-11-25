"""Edge-case file audit: craft tricky files and record what loading + chunking does with each.
Usage: python audit_files.py <workdir>"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tests"))
from conftest import make_pdf  # noqa: E402

from rag.chunking import chunk_pages  # noqa: E402
from rag.loaders import load_documents  # noqa: E402

W = Path(sys.argv[1])
shutil.rmtree(W, ignore_errors=True)

CASES: dict[str, tuple[str, bytes, str]] = {
    # name: (filename, bytes, what we expect)
    "setext headings": ("setext.md", "Title\n=====\n\nIntro.\n\nPart One\n--------\nAlpha text.\n".encode(),
                        "sections 'Title' and 'Title > Part One'"),
    "YAML front matter": ("front.md", "---\ntitle: Notes\ntags: [a, b]\n---\n# Notes\n\nBody text.\n".encode(),
                          "front matter not treated as content/headings"),
    "no headings": ("plain.md", ("Just paragraphs. " * 300).encode(), "1 section, several chunks"),
    "only H2, no H1": ("h2only.md", "## Setup\nInstall it.\n\n## Usage\nRun it.\n".encode(),
                       "sections 'Setup' and 'Usage' (no title)"),
    "deep nesting H1-H6": ("deep.md", "# A\n## B\n### C\n#### D\n##### E\n###### F\nDeep text.\n".encode(),
                           "path 'A > B > C > D > E > F'"),
    "HTML + table": ("table.md", "# Prices\n\n<div align='center'><b>Note</b></div>\n\n| Plan | Price |\n|---|---|\n"
                     "| Basic | $5 |\n| Pro | $20 |\n".encode(), "table rows kept together in text"),
    "cp1252 encoding": ("legacy.txt", "Caf\xe9 \x93smart quotes\x94 \x96 price \x80 5".encode("latin-1"),
                        "accented/smart-quote characters readable"),
    "UTF-8 BOM": ("bom.md", "﻿# Title\n\nBody.\n".encode("utf-8"), "first heading still detected"),
    "CRLF line endings": ("crlf.md", b"# Title\r\n\r\n## Sub\r\nBody.\r\n", "headings detected"),
    "Arabic file name": ("ملاحظات.md", "# ملاحظات\n\nنص عربي للاختبار.\n".encode(), "loads, readable name"),
    "same name, two folders A": ("a/notes.md", b"# Notes A\n\nAlpha.\n", "both kept, distinct ids"),
    "same name, two folders B": ("b/notes.md", b"# Notes B\n\nBravo.\n", "both kept, distinct ids"),
    "minified, no spaces": ("minified.txt", ("x" * 20000).encode(), "chunked without hanging"),
    "huge section": ("huge.md", ("# Big\n\n" + "Sentence with words. " * 5000).encode(), "many chunks"),
    "whitespace only": ("blank.md", b"   \n\n  ", "reported as skipped"),
    "headings only": ("heads.md", b"# A\n## B\n", "reported as skipped"),
    "unsupported .rst": ("guide.rst", b"Title\n=====\n\nText.", "reported or at least not crash"),
    "unsupported .docx": ("report.docx", b"PK\x03\x04fake", "reported or at least not crash"),
    "unsupported .html": ("page.html", b"<h1>Hi</h1><p>Text</p>", "reported or at least not crash"),
    "uppercase extension": ("UPPER.MD", b"# Upper\n\nText.\n", "loaded like .md"),
    "hidden/temp file": ("~$draft.md", b"# Draft\n\nTemp.\n", "ideally ignored (Office lock file)"),
}

for fname, data, _ in CASES.values():
    p = W / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
make_pdf(W / "scan.pdf", [])
make_pdf(W / "text.pdf", ["Quarterly revenue grew 12%.", "Costs fell (slightly) by 3%."])

skipped: list[str] = []
pages = load_documents(W, skipped=skipped)
chunks = chunk_pages(pages, 1000, 150)
by_source: dict[str, list] = {}
for pg in pages:
    by_source.setdefault(pg.source, []).append(pg)
chunk_count: dict[str, int] = {}
for c in chunks:
    chunk_count[c.source] = chunk_count.get(c.source, 0) + 1

print(f"{'case':<26} {'result'}")
for name, (fname, _, expect) in CASES.items():
    src = fname
    got = by_source.get(src)
    if got is None:
        status = "SKIPPED (reported)" if src in skipped else "IGNORED SILENTLY"
        detail = ""
    else:
        status = f"{len(got)} section(s), {chunk_count.get(src, 0)} chunk(s)"
        detail = " | ".join(f"[{g.section or '-'}] {g.text[:45]!r}" for g in got[:3])
    print(f"{name:<26} {status:<24} expect: {expect}\n{'':<26} {detail}")
print("\nscan.pdf:", "skipped (reported)" if "scan.pdf" in skipped else "loaded?")
print("text.pdf:", [p.text for p in by_source.get("text.pdf", [])])
print("chunk ids unique:", len({c.id for c in chunks}) == len(chunks))
