"""Build a generalization corpus from third-party docs shipped inside installed packages.
Usage: python build_corpus.py <site-packages> <out-dir>"""

import shutil
import sys
from pathlib import Path

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)


def description(pkg_prefix: str) -> str:
    dist = next(SP.glob(f"{pkg_prefix}-*.dist-info"))
    text = (dist / "METADATA").read_text(encoding="utf-8")
    header, _, body = text.partition("\n\n")  # METADATA headers, blank line, long description
    return body.strip()


def make_pdf(path: Path, text: str, lines_per_page: int = 48, width: int = 95) -> int:
    """Multi-page PDF with a real text layer (Helvetica, latin-1)."""
    lines: list[str] = []
    for raw in text.splitlines():
        raw = raw.encode("latin-1", "replace").decode("latin-1")
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            cut = cut if cut > 20 else width
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(raw)
    pages = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    n_pages = len(pages)
    # 1 catalog, 2 pages, 3 font, then (page, content) pairs
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, page in enumerate(pages):
        content = "BT /F1 9 Tf 40 760 Td 14 TL " + " ".join(f"({esc(line)}) '" for line in page) + " ET"
        data = content.encode("latin-1")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                       f"/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objects.append(b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)
    return n_pages


for prefix, name in [("pydantic", "pydantic.md"), ("python_dotenv", "python-dotenv.md"),
                     ("charset_normalizer", "charset-normalizer.md"), ("orjson", "orjson.md"),
                     ("typer", "typer.md"), ("annotated_types", "annotated-types.md")]:
    (OUT / name).write_text(description(prefix), encoding="utf-8")

(OUT / "tqdm-manual.txt").write_text(description("tqdm"), encoding="utf-8")
numpy_license = next(SP.glob("numpy-*.dist-info")) / "licenses" / "LICENSE.txt"
shutil.copy(numpy_license, OUT / "numpy-licenses.txt")
pages = make_pdf(OUT / "httpcore.pdf", description("httpcore"))

for f in sorted(OUT.iterdir()):
    print(f"{f.name:<24} {f.stat().st_size / 1024:7.1f} KB" + (f"  ({pages} pages)" if f.suffix == ".pdf" else ""))
