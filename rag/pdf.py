"""Recover document structure from PDFs.

PDF text extraction returns a flat stream of lines per page, so PDF passages had no headings and
none of the context that makes Markdown retrieval work. Real PDFs do carry structure, just not as
markup: headings are set in a larger or bold font. This module reads the font of every line,
learns the body size of each document, turns larger/bold short lines into headings, drops running
headers and footers, and returns sections that continue across page breaks (with page numbers
kept for citations).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from pypdf import PdfReader

HEADING_SIZE_RATIO = 1.15  # a line this much larger than body text is a heading candidate
MAX_HEADING_CHARS = 120
MAX_HEADING_SHARE = 0.4  # if more lines than this look like headings, the "body" guess is wrong
MAX_RUNNING_HEADER_CHARS = 60
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+[A-Z]")


@dataclass
class Line:
    page: int
    text: str
    size: float
    bold: bool
    level: int = 0  # 0 = body text, 1..6 = heading level


def _page_lines(page, number: int) -> list[Line]:
    """Group text runs into lines (same baseline), keeping the dominant font size and weight."""
    runs: list[tuple[float, str, float, bool]] = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text:
            return
        scale = abs(tm[3] or 1) * abs(cm[3] or 1)
        y = tm[5] * (cm[3] or 1) + cm[5]
        font = str(font_dict.get("/BaseFont", "")) if font_dict else ""
        bold = any(w in font.lower() for w in ("bold", "black", "heavy", "semibold"))
        runs.append((y, text, (font_size or 0) * scale, bold))

    page.extract_text(visitor_text=visit)
    lines: list[Line] = []
    current: list[tuple[str, float, bool]] = []
    last_y = None

    def flush():
        text = "".join(t for t, _, _ in current).strip()
        if text:
            weights: Counter = Counter()
            bold_chars = 0
            for t, s, b in current:
                weights[round(s * 2) / 2] += len(t.strip())
                bold_chars += len(t.strip()) if b else 0
            size = weights.most_common(1)[0][0] if weights else 0.0
            lines.append(Line(number, text, size, bold_chars > sum(weights.values()) / 2))
        current.clear()

    for y, text, size, bold in runs:
        if last_y is not None and abs(y - last_y) > 2:
            flush()
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                flush()
            current.append((part, size, bold))
        last_y = y
    flush()
    return lines


def _normalize(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def remove_running_headers(pages: list[list[Line]]) -> list[list[Line]]:
    """Drop lines that repeat at the top or bottom of most pages ("Page 3", a document title)."""
    if len(pages) < 3:
        return pages
    counts: Counter = Counter()
    for lines in pages:
        edge = {_normalize(line.text) for line in lines[:2] + lines[-2:]}
        counts.update(edge)
    # Only short lines: running headers/footers are short, while a body sentence that happens to
    # repeat near the edge of every short page must survive (an early version deleted it).
    repeated = {text for text, n in counts.items()
                if n >= max(3, len(pages) // 2) and len(text) <= MAX_RUNNING_HEADER_CHARS and len(text.split()) <= 10}
    return [[line for line in lines if _normalize(line.text) not in repeated] for lines in pages]


def _looks_like_heading(line: Line) -> bool:
    text = line.text
    return (2 <= len(text) <= MAX_HEADING_CHARS and any(c.isalpha() for c in text)
            and not text.endswith((".", ",", ";", ":")) and len(text.split()) <= 15)


def assign_heading_levels(lines: list[Line]) -> str | None:
    """Mark headings by font (larger or bold than the body text); fall back to numbered headings
    ("2.1 Setup") for single-font PDFs. Levels follow font size: the largest is level 1.
    Returns how headings were found: "font", "numbered" or None."""
    if not lines:
        return None
    by_size: Counter = Counter()
    for line in lines:
        by_size[line.size] += len(line.text)
    body = by_size.most_common(1)[0][0]
    larger = [ln for ln in lines if ln.size >= body * HEADING_SIZE_RATIO and _looks_like_heading(ln)]
    bold = [ln for ln in lines if ln.bold and abs(ln.size - body) < body * 0.1 and _looks_like_heading(ln)]
    candidates = larger + bold
    if candidates and len(candidates) <= MAX_HEADING_SHARE * len(lines):
        sizes = sorted({ln.size for ln in larger}, reverse=True)
        for ln in larger:
            ln.level = min(sizes.index(ln.size) + 1, 6)
        for ln in bold:
            ln.level = min(len(sizes) + 1, 6)
        return "font"
    numbered = [ln for ln in lines if _NUMBERED.match(ln.text) and _looks_like_heading(ln) and len(ln.text) <= 80]
    if 2 <= len(numbered) <= MAX_HEADING_SHARE * len(lines):
        for ln in numbered:
            ln.level = min(_NUMBERED.match(ln.text).group(1).count(".") + 1, 6)
        return "numbered"
    return None


def sections(lines: list[Line], use_title: bool = True) -> list[tuple[int, str, str]]:
    """(page, heading path, body) pieces. A section continues across pages; each page of it is its
    own piece so citations keep page numbers. With font headings, the first level-1 heading is the
    document title; numbered headings ("1 Introduction") never are."""
    out: list[tuple[int, str, str]] = []
    title: str | None = None
    stack: list[tuple[int, str]] = []
    path, buffer, page = "", [], lines[0].page if lines else 1

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            out.append((page, path, body))
        buffer.clear()

    for line in lines:
        if line.page != page:
            flush()
            page = line.page
        if not line.level:
            buffer.append(line.text)
            continue
        flush()
        if use_title and title is None and line.level == 1 and not out and not stack:
            title, path = line.text, line.text
            continue
        while stack and stack[-1][0] >= line.level:
            stack.pop()
        stack.append((line.level, line.text))
        path = " > ".join(([title] if title else []) + [h for _, h in stack])
    flush()
    return out


def read_pdf(path) -> list[tuple[int, str, str]]:
    """Structured pieces of a PDF: (page number, heading path, text). Heading path is "" when the
    PDF has no detectable headings (then each page is one piece, as before)."""
    reader = PdfReader(path)
    pages = [_page_lines(page, n) for n, page in enumerate(reader.pages, start=1)]
    pages = remove_running_headers(pages)
    lines = [line for page_lines in pages for line in page_lines]
    found_by = assign_heading_levels(lines)
    return sections(lines, use_title=found_by == "font")
