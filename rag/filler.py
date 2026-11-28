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
    r"|not (?:mentioned|provided|listed|specified|applicable)|nothing (?:is )?(?:listed|mentioned)"
    r"|^\s*[-*]\s*\[[^\]\d][^\]]*\]\s*$",  # a placeholder bullet: "- [Not applicable for this version]"
    re.IGNORECASE | re.MULTILINE,
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


def insert_sections(answer: str, new: list[tuple[str, str]], order: list[str], same, before: str | None = None) -> str:
    """Add sections ("### name" + body) in document order: each goes before the first existing
    section that comes later in `order`, else before the `before` heading (a closing note), else at
    the end. `same(a, b)` decides whether a heading names an item."""
    rank = {name: i for i, name in enumerate(order)}

    def position(heading: str) -> int | None:
        return next((rank[name] for name in order if same(name, heading)), None)

    for name, body in sorted(new, key=lambda nb: rank[nb[0]], reverse=True):
        block = f"### {name}\n{body.strip()}\n\n"
        found = list(_HEADING.finditer(answer))
        later = next((m for m in found if (pos := position(m.group(1))) is not None and pos > rank[name]), None)
        if later is None and before:
            later = next((m for m in found if m.group(1).strip() == before), None)
        if later is None:
            answer = answer.rstrip() + "\n\n" + block.rstrip() + "\n"
        else:
            answer = answer[:later.start()] + block + answer[later.start():]
    return answer


def splice(answer: str, bodies: dict[str, str]) -> str:
    """Replace the bodies of the sections named in `bodies`, keeping everything else as it was."""
    out, last = [], 0
    for heading, start, end in sections(answer):
        if heading in bodies:
            trailing = answer[start:end][len(answer[start:end].rstrip()):]
            out += [answer[last:start], "\n" + bodies[heading].strip(), trailing or "\n"]
            last = end
    return "".join(out) + answer[last:]
