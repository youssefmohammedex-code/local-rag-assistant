"""Detect a model that has started repeating itself, so its answer can be stopped early.

A 7B model occasionally falls into a loop and never stops on its own: one benchmark answer ran
for 8.8 hours (345,000 characters), cycling through the same 4 sections with citation numbers
counting up ([2080], [2081], ...). The answer-length cap ends such an answer eventually, but at
about 3 tokens/s on a laptop CPU that is still ~15 minutes of repeated text.

The rule: once citation numbers are removed, the last lines are one block of 1-30 lines repeated
3 times in a row. Normal answers repeat a line now and then ("- No changes listed" under several
releases), but always with different headings in between, so they never form such a block.
"""

from __future__ import annotations

import re

REPEATS = 3  # copies of the block in a row that count as a loop
MAX_BLOCK_LINES = 30  # longest repeated block looked for (a few whole sections)
MIN_BLOCK_CHARS = 40  # shorter blocks ("}", "---", "|---|") repeat legitimately

_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


class LoopDetector:
    """Feed streamed text with `feed`; it returns True once the text has started looping."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._partial = ""

    def feed(self, text: str) -> bool:
        self._partial += text
        *complete, self._partial = self._partial.split("\n")
        return any(self._add(line) for line in complete)

    def _add(self, line: str) -> bool:
        line = _CITATION.sub("", line).strip()
        if not line:
            return False
        lines = self._lines
        lines.append(line)
        del lines[:-REPEATS * MAX_BLOCK_LINES]
        for size in range(1, MAX_BLOCK_LINES + 1):
            span = size * REPEATS
            if span > len(lines):
                break
            tail = lines[-span:]
            if all(tail[i] == tail[i + size] for i in range(span - size)) and \
                    sum(len(x) for x in tail[:size]) >= MIN_BLOCK_CHARS:
                return True
        return False
