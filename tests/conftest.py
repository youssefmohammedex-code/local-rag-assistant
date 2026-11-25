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
