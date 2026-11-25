"""Load PDF, Markdown and plain-text files into page/section-level records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NOTION_ID = re.compile(r"\s+[0-9a-f]{32}$")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*(\r?\n|\Z)", re.DOTALL)
_SETEXT = re.compile(r" {0,3}(={2,}|-{3,})\s*")


def is_ignored(path: Path, root: Path) -> bool:
    """Hidden files/folders (.obsidian, .git) and Office lock files (~$report.docx)."""
    return any(part.startswith((".", "~$")) for part in path.relative_to(root).parts)


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


def read_text_file(path: Path) -> str:
    """UTF-8 (with or without BOM), falling back to Windows-1252 instead of silently dropping
    bytes: legacy Windows files would otherwise lose every accented letter and smart quote."""
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_front_matter(text: str) -> str:
    """Remove a leading YAML front-matter block (Obsidian, Jekyll, Hugo) so metadata isn't indexed."""
    match = _FRONT_MATTER.match(text)
    return text[match.end():] if match else text


def setext_to_atx(lines: list[str]) -> list[str]:
    """Convert setext headings ("Title" underlined with === or ---) to "# Title" / "## Title".
    Only a single-line paragraph directly above the underline counts, so a "---" rule after a
    longer paragraph stays a rule."""
    out: list[str] = []
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        prev = out[-1] if out else ""
        if (not in_code and _SETEXT.fullmatch(line) and prev.strip()
                and not prev.lstrip().startswith(("#", "|", "-", "*", ">", "```"))
                and (len(out) < 2 or not out[-2].strip())):
            out[-1] = ("# " if "=" in line else "## ") + prev.strip()
            continue
        out.append(line)
    return out


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

    for line in setext_to_atx(strip_front_matter(text.lstrip("﻿")).splitlines()):
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


def load_documents(docs_dir: Path, skipped: list[str] | None = None,
                   unsupported: list[str] | None = None) -> list[Page]:
    """Load every supported file. Files that yield no text (empty files, scanned PDFs without a
    text layer, unreadable PDFs) are appended to `skipped`, and files of other types (.docx,
    .html, ...) to `unsupported`, instead of disappearing silently."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    pages: list[Page] = []
    for path in sorted(docs_dir.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or is_ignored(path, docs_dir):
            continue
        if suffix not in SUPPORTED_SUFFIXES:
            if unsupported is not None:
                unsupported.append(path.relative_to(docs_dir).as_posix())
            continue
        source = path.relative_to(docs_dir).as_posix()
        before = len(pages)
        if suffix == ".pdf":
            try:
                reader = PdfReader(path)
                for number, pdf_page in enumerate(reader.pages, start=1):
                    text = pdf_page.extract_text() or ""
                    if text.strip():
                        pages.append(Page(source, number, text))
            except Exception:  # corrupt or encrypted PDF: skip it, report it
                pass
        elif suffix == ".md":
            text = read_text_file(path)
            for part, (section, body) in enumerate(split_markdown(text), start=1):
                pages.append(Page(source, 1, body, section=section, part=part))
        else:
            text = read_text_file(path)
            if text.strip():
                pages.append(Page(source, 1, text))
        if len(pages) == before and skipped is not None:
            skipped.append(source)
    return pages
