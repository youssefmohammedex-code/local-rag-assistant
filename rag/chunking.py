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


MIN_SECTION_SHARE = 0.4  # sections shorter than this share of chunk_size are merged with what follows


def _merge(group: list[Page]) -> Page:
    """One page from consecutive small sections: the shared heading path becomes the section, and
    each section's own remaining heading is kept inline above its text."""
    if len(group) == 1:
        return group[0]
    paths = [g.section.split(" > ") if g.section else [] for g in group]
    common: list[str] = []
    for level in zip(*paths, strict=False):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    parts = []
    for page, path in zip(group, paths, strict=True):
        rel = " > ".join(path[len(common):])
        parts.append(f"{rel}\n{page.text}" if rel else page.text)
    return Page(group[0].source, group[0].page, "\n\n".join(parts), section=" > ".join(common), part=group[0].part)


def pack_small_sections(pages: list[Page], min_chars: int, max_chars: int) -> list[Page]:
    """Merge runs of tiny sections ("Requirements: Python 3.8+") with the sections that follow,
    up to `max_chars`. A 20-character passage is a weak match next to longer passages that merely
    mention the same words; measured on PDFs and changelogs, merging restores the lost context."""
    out: list[Page] = []
    group: list[Page] = []
    size = 0
    for page in pages:
        if group and (page.source != group[0].source or size >= min_chars or size + len(page.text) > max_chars):
            out.append(_merge(group))
            group, size = [], 0
        group.append(page)
        size += len(page.text)
    if group:
        out.append(_merge(group))
    return out


SMALL_SECTIONS = "lookahead"  # "lookahead" | "pack" | "off" (kept switchable for benchmarking)


def with_lookahead(pages: list[Page], i: int, min_chars: int, max_chars: int) -> str:
    """A tiny section's text followed by the sections after it (with their headings), until it
    reaches `min_chars`. The section keeps its own heading and metadata, unlike merging."""
    page = pages[i]
    text = page.text
    j = i + 1
    while len(text) < min_chars and j < len(pages) and pages[j].source == page.source:
        nxt = pages[j]
        heading = nxt.section.split(" > ")[-1] if nxt.section else ""
        text += f"\n\n{heading}\n{nxt.text}" if heading else f"\n\n{nxt.text}"
        j += 1
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return text


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int, small: str | None = None) -> list[Chunk]:
    """Chunk every page. Markdown/PDF chunks are prefixed with their heading path ("contextual
    headers"), so a passage from "Module C — Dispatch" still says so after it is cut out.
    Tiny sections ("Requirements: Python 3.8+") borrow context from what follows them."""
    small = small or SMALL_SECTIONS
    min_chars = int(chunk_size * MIN_SECTION_SHARE)
    if small == "pack":
        pages = pack_small_sections(pages, min_chars, chunk_size)
    chunks: list[Chunk] = []
    for index, page in enumerate(pages):
        if small == "lookahead" and len(page.text) < min_chars:
            # Exactly one passage: splitting would leave a leftover made only of borrowed text.
            pieces = [normalize(with_lookahead(pages, index, min_chars, chunk_size))]
        else:
            pieces = split_text(page.text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            text = f"{page.section}\n\n{piece}" if page.section else piece
            chunk_id = f"{page.source}::p{page.page}::s{page.part}::c{i}"
            chunks.append(Chunk(chunk_id, text, page.source, page.page, page.section))
    return chunks
