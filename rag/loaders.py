"""Load PDF, Markdown and plain-text files into page-level records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


@dataclass(frozen=True)
class Page:
    source: str  # path relative to the docs folder, e.g. "regulations/handbook.pdf"
    page: int  # 1-based page number (always 1 for text files)
    text: str


def load_documents(docs_dir: Path) -> list[Page]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    pages: list[Page] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        source = path.relative_to(docs_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            for number, pdf_page in enumerate(reader.pages, start=1):
                text = pdf_page.extract_text() or ""
                if text.strip():
                    pages.append(Page(source, number, text))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append(Page(source, 1, text))
    return pages
