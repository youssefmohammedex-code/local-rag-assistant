"""Build realistic test PDFs from package READMEs: headings in larger bold fonts, body text,
monospace code, a running header and a page-number footer on every page (like real documents).
Also builds one "plain" PDF with a single font and no styling, the hardest case.

Usage: python make_pdfs.py <site-packages> <out-dir>"""

import re
import shutil
import sys
from pathlib import Path

SP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)

FONTS = {"F1": "Helvetica", "F2": "Helvetica-Bold", "F3": "Courier"}
STYLES = {1: ("F2", 18), 2: ("F2", 14), 3: ("F2", 12), "body": ("F1", 10), "code": ("F3", 9)}


def description(prefix: str) -> str:
    dist = next(SP.glob(f"{prefix}-*.dist-info"))
    return (dist / "METADATA").read_text(encoding="utf-8").partition("\n\n")[2]


def clean(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images/badges
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # links -> text
    line = line.replace("**", "").replace("`", "'")
    return line.encode("latin-1", "replace").decode("latin-1")


def styled_lines(markdown: str) -> list[tuple[str, str]]:
    """(style, text) lines. Headings keep no '#'; code blocks use the code style."""
    out, in_code = [], False
    for raw in markdown.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", raw)
        if m and not in_code:
            out.append((min(len(m.group(1)), 3), clean(m.group(2)).strip()))
        else:
            out.append(("code" if in_code else "body", clean(raw)))
    return out


def wrap(text: str, width: int) -> list[str]:
    lines = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        cut = cut if cut > 20 else width
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    return lines + [text]


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(path: Path, lines: list[tuple[object, str]], header: str | None, plain: bool = False) -> int:
    pages, page, y = [], [], 740
    for style, text in lines:
        font, size = STYLES["body"] if plain else STYLES[style]
        gap = size + (8 if isinstance(style, int) and not plain else 4)
        for part in wrap(text, 95 if size <= 10 else 70) if text else [""]:
            if y - gap < 60:
                pages.append(page)
                page, y = [], 740
            y -= gap
            page.append((font, size, y, part))
    pages.append(page)

    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    font_ids = {}
    for name, base in FONTS.items():
        objects.append(f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} >>".encode())
        font_ids[name] = len(objects)
    font_res = " ".join(f"/{n} {i} 0 R" for n, i in font_ids.items())
    kids = []
    for n, items in enumerate(pages, 1):
        ops = []
        if header:  # running header + footer, as in most real documents
            ops.append(f"BT /F1 8 Tf 40 770 Td ({esc(header)}) Tj ET")
            ops.append(f"BT /F1 8 Tf 290 30 Td (Page {n}) Tj ET")
        for font, size, y, text in items:
            ops.append(f"BT /{font} {size} Tf 40 {y} Td ({esc(text)}) Tj ET")
        data = "\n".join(ops).encode("latin-1")
        objects.append(b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                       f"/Resources << /Font << {font_res} >> >> >>".encode())
        kids.append(len(objects))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)
    return len(pages)


if __name__ == "__main__":
    for prefix, name, header in [("httpcore", "httpcore-guide.pdf", "HTTP Core - documentation"),
                                 ("rich", "rich-guide.pdf", "Rich - README"),
                                 ("pandas", "pandas-guide.pdf", "pandas - README")]:
        n = build_pdf(OUT / name, styled_lines(description(prefix)), header)
        print(f"{name:<24} styled, {n} pages")
    n = build_pdf(OUT / "fsspec-plain.pdf", styled_lines(description("fsspec")), None, plain=True)
    print(f"{'fsspec-plain.pdf':<24} single font, no styling, {n} pages")
