import dataclasses
import hashlib
from pathlib import Path

import pytest

from rag.config import Settings
from rag.pipeline import RAGPipeline


def make_pdf(path: Path, lines: list[str]) -> None:
    """Write a minimal one-page PDF. With no lines it has no text layer, like a scanned document."""
    content = "BT /F1 12 Tf 72 720 Td 16 TL " + " ".join(f"({line}) '" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    path.write_bytes(out.encode("latin-1"))


def make_styled_pdf(path: Path, pages: list[list[tuple[str, int, str]]], header: str | None = None) -> None:
    """Multi-page PDF where each line is (font, size, text); font is "regular", "bold" or "mono".
    `header` adds a running header and a "Page N" footer to every page, like real documents."""
    fonts = {"regular": ("F1", "Helvetica"), "bold": ("F2", "Helvetica-Bold"), "mono": ("F3", "Courier")}

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    font_ids = {}
    for name, base in fonts.values():
        objects.append(f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} >>".encode())
        font_ids[name] = len(objects)
    resources = " ".join(f"/{n} {i} 0 R" for n, i in font_ids.items())
    kids = []
    for number, lines in enumerate(pages, 1):
        ops, y = [], 740
        if header:
            ops += [f"BT /F1 8 Tf 40 770 Td ({esc(header)}) Tj ET", f"BT /F1 8 Tf 290 30 Td (Page {number}) Tj ET"]
        for font, size, text in lines:
            y -= size + 6
            ops.append(f"BT /{fonts[font][0]} {size} Tf 40 {y} Td ({esc(text)}) Tj ET")
        data = "\n".join(ops).encode("latin-1")
        objects.append(b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R "
                       f"/Resources << /Font << {resources} >> >> >>".encode())
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


def fake_embed(texts: list[str], kind: str = "document") -> list[list[float]]:
    """Deterministic 8-dim vectors so tests run without Ollama."""
    return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


@pytest.fixture
def pipeline(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    p = RAGPipeline(dataclasses.replace(Settings(), docs_dir=docs, db_dir=tmp_path / "db"))
    p.store.embed = fake_embed
    return p
