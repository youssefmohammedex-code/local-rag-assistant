"""Find and replace filler sections in list answers.

Asked only for names ("which releases does the changelog list?"), qwen2.5:7b lists every section
but writes a filler bullet under each one ("No specific versions mentioned", "not relevant to the
question") although its passages contain that section's content. Two rewordings of the answer
prompt didn't stop it on a changelog they weren't tuned on: the model checks each section for
the thing the question names, and finds none. So the answer is left as written and only its
filler sections are regenerated afterwards, without the question (see RAGPipeline.repair_filler).
"""

from __future__ import annotations

import re

FILLER = re.compile(
    r"not relevant|no relevant|\bno (?:specific |further |additional )?\w+(?: \w+)? (?:are |is )?"
    r"(?:listed|mentioned|provided|given|specified)|no (?:specific )?(?:information|details|changes)"
    r"|not (?:mentioned|provided|listed|specified)|nothing (?:is )?(?:listed|mentioned)",
    re.IGNORECASE,
)
MAX_FILLER_CHARS = 160  # a section this short that says "nothing here" is filler, not content

_HEADING = re.compile(r"^#{2,4} +(.+?) *$", re.MULTILINE)


def sections(answer: str) -> list[tuple[str, int, int]]:
    """(heading, body start, body end) for every "## / ### / ####" section of an answer."""
    found = list(_HEADING.finditer(answer))
    return [(m.group(1).strip(), m.end(), found[i + 1].start() if i + 1 < len(found) else len(answer))
            for i, m in enumerate(found)]


def is_filler(body: str) -> bool:
    """A section body that says there is nothing there, or has no text at all."""
    text = body.strip()
    return not re.sub(r"[\W\d_]", "", re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)) or \
        (len(text) <= MAX_FILLER_CHARS and bool(FILLER.search(text)))


def filler_headings(answer: str) -> list[str]:
    return [heading for heading, start, end in sections(answer) if is_filler(answer[start:end])]


def splice(answer: str, bodies: dict[str, str]) -> str:
    """Replace the bodies of the sections named in `bodies`, keeping everything else as it was."""
    out, last = [], 0
    for heading, start, end in sections(answer):
        if heading in bodies:
            trailing = answer[start:end][len(answer[start:end].rstrip()):]
            out += [answer[last:start], "\n" + bodies[heading].strip(), trailing or "\n"]
            last = end
    return "".join(out) + answer[last:]
