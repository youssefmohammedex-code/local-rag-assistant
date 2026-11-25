"""Load PDF, Markdown and plain-text files into page/section-level records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NOTION_ID = re.compile(r"\s+[0-9a-f]{32}$")


@dataclass(frozen=True)
class Page:
    source: str  # path relative to the docs folder, e.g. "regulations/handbook.pdf"
    page: int  # 1-based page number (always 1 for text and Markdown files)
    text: str
    section: str = ""  # heading path for Markdown, e.g. "Guide > 7. Modules > Module C"
    part: int = 0  # section index within the file, keeps chunk ids unique


def pretty_source(source: str) -> str:
    """'notes/Arch v2 421ffbd...e2.md' -> 'Arch v2' (drops folder, extension and Notion export ids)."""
    stem = Path(source).stem
    return _NOTION_ID.sub("", stem).strip() or stem


def split_markdown(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading_path, body) sections. The first H1 is treated as the document
    title and prefixed to every path. Headings inside fenced code blocks are ignored."""
    sections: list[tuple[str, str]] = []
    title: str | None = None
    stack: list[tuple[int, str]] = []
    path, buffer, in_code = "", [], False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((path, body))
        buffer.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if not match:
            buffer.append(line)
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2).strip()
        if title is None and level == 1 and not sections:
            title, path = heading, heading
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        path = " > ".join(([title] if title else []) + [h for _, h in stack])
    flush()
    return sections


def load_documents(docs_dir: Path) -> list[Page]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    pages: list[Page] = []
    for path in sorted(docs_dir.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in SUPPORTED_SUFFIXES:
            continue
        source = path.relative_to(docs_dir).as_posix()
        if suffix == ".pdf":
            reader = PdfReader(path)
            for number, pdf_page in enumerate(reader.pages, start=1):
                text = pdf_page.extract_text() or ""
                if text.strip():
                    pages.append(Page(source, number, text))
        elif suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for part, (section, body) in enumerate(split_markdown(text), start=1):
                pages.append(Page(source, 1, body, section=section, part=part))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append(Page(source, 1, text))
    return pages
