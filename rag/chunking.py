"""Split text into overlapping chunks that prefer to break at natural boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.loaders import Page

# Tried in order: paragraph, line, sentence, word.
_BOUNDARIES = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    section: str = ""


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window of at most `chunk_size` characters, cut at the best boundary
    in the second half of the window, with `overlap` characters carried over."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = normalize(text)
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            for boundary in _BOUNDARIES:
                cut = window.rfind(boundary)
                if cut > chunk_size // 2:
                    end = start + cut + len(boundary)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # Step back by `overlap`, then forward to the next word start so chunks don't begin mid-word.
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        next_start = space + 1 if overlap and space != -1 else next_start
        start = max(next_start, start + 1)
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every page. Markdown chunks are prefixed with their heading path ("contextual
    headers"), so a passage from "Module C — Dispatch" still says so after it is cut out."""
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            text = f"{page.section}\n\n{piece}" if page.section else piece
            chunk_id = f"{page.source}::p{page.page}::s{page.part}::c{i}"
            chunks.append(Chunk(chunk_id, text, page.source, page.page, page.section))
    return chunks
