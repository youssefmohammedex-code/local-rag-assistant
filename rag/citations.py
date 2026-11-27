"""Check an answer's citations against the passages it was given, without another model call.

The 7B model cites by position as often as by content (it put [1], [2], [3] under the 1st, 2nd and
3rd headings whatever those passages said), so a citation is a claim to verify, not a fact. Each
cited sentence is compared with every passage by word overlap: the share of the sentence's content
words that the passage contains. That tells three cases apart:

- supported:   a cited passage contains most of the sentence's words
- miscited:    a cited passage doesn't, but another passage clearly does (the likely right id)
- unsupported: no passage does (the sentence may be invented or a paraphrase too loose to check)

Two rules make the overlap stricter where the model most often goes wrong:

- Numbers and codes must match. "Requires Python 3.6" is not supported by a passage that says 3.8,
  however many other words they share; a pull-request number (#478) must be in the passage.
- Only a passage's own section counts. A passage can run on into the next sections (the 0.14.6
  release's passage ends with "0.14.5 (January 18th, 2022)" and that release's changes), and the
  model moves facts between neighbouring headings. With `headings` (the documents' section
  names), a passage is cut at the first line that is another section's heading, so a 0.14.5
  change cited for 0.14.6 is caught.

Word overlap can't judge meaning: a sentence that reuses a passage's words but reverses what it
says still counts as supported. It catches wrong ids and sentences with no basis in the passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.bm25 import identifiers

SUPPORTED = 0.5  # share of a sentence's content words a passage must contain to support it
MISCITED_MARGIN = 0.2  # how much better another passage must be to call a citation wrong
MIN_WORDS = 3  # sentences with fewer content words ("Yes [2].") are too short to judge

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_WORD = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|[a-z0-9_]+(?:[.'][a-z0-9_]+)*")  # 3,000 is one number
_STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its itself just let may me might more most must my no nor not
now of off on once only or other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your
""".split())


@dataclass
class CitationCheck:
    claim: str  # the sentence, without its citations
    cited: list[int]  # passage ids it cites (1-based)
    status: str  # "supported", "miscited" or "unsupported"
    score: float  # best overlap among the cited passages
    best: int | None  # the passage that supports it best, when that is not a cited one
    best_score: float


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


# "Module C", "Option B", "2 weeks"; not the digits inside a longer number like 2.0 or 3,000
_SINGLE_CODE = re.compile(r"(?<=[\w,;:)] )[A-Z](?![\w'])|(?<![\w.,])\d(?![\w]|[.,]\d)")


def content_words(text: str) -> set[str]:
    """Lower-cased, lightly stemmed words minus stopwords. Single characters are dropped, except
    digits and capital letters inside a sentence, which are kept as codes ("Module C")."""
    words = {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}
    return words | {m.group().lower() for m in _SINGLE_CODE.finditer(text)}


def _is_code(word: str) -> bool:
    return len(word) == 1 or any(c.isdigit() for c in word)


def claims(answer: str) -> list[tuple[str, list[int], str | None]]:
    """Cited sentences of an answer: (sentence without citations, cited ids, the markdown heading it
    is under or None). A citation covers the text since the previous citation or sentence end; text
    with no citation is not a claim."""
    found = []
    heading = None
    for line in answer.splitlines():
        if match := re.match(r"^\s*#{1,6}\s+(.+)$", line):
            heading = _CITATION.sub("", match.group(1)).strip(" *")
            continue
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)  # bullet or list number
        start = 0
        for match in _CITATION.finditer(line):
            text = line[start:match.start()]
            text = re.split(r"(?<=[.!?])\s+(?=[A-Z`*])", text.strip())[-1] if text.strip() else ""
            ids = [int(n) for n in re.findall(r"\d+", match.group(1))]
            if text.strip(" .,;:*") and ids:
                found.append((text.strip(), ids, heading))
            elif ids and found and not text.strip(" .,;:*"):
                found[-1] = (found[-1][0], found[-1][1] + ids, found[-1][2])  # "[1][3]" or "[1] [3]"
            start = match.end()
    return found


def overlap(claim_words: set[str], passage_words: set[str]) -> float:
    """Share of the claim's words in the passage; below the support threshold if any number or
    code in the claim is missing from the passage."""
    if not claim_words:
        return 0.0
    score = len(claim_words & passage_words) / len(claim_words)
    codes = {w for w in claim_words if _is_code(w)}
    if codes - passage_words:
        score = min(score, SUPPORTED - 0.01)
    return score


def summary(checks: list[CitationCheck]) -> str:
    """One line for the UI and CLI, e.g. "12 of 14 cited sentences match their passage · 2 to check"."""
    if not checks:
        return ""
    ok = sum(c.status == "supported" for c in checks)
    issues = len(checks) - ok
    return f"{ok} of {len(checks)} cited sentences match their passage" + (f" · {issues} to check" if issues else "")


def issues(checks: list[CitationCheck]) -> list[str]:
    """A readable line per citation that didn't check out."""
    lines = []
    for c in checks:
        claim = c.claim if len(c.claim) <= 110 else c.claim[:107] + "…"
        cited = ", ".join(map(str, c.cited))
        if c.status == "miscited":
            lines.append(f"“{claim}” cites [{cited}], but passage [{c.best}] matches it")
        elif c.status == "unsupported":
            lines.append(f"“{claim}” cites [{cited}], and no passage clearly supports it")
    return lines


def segments(passage: str, section: str, paths: set[str]) -> list[tuple[str | None, str]]:
    """Split a passage into (heading, text) parts: first its own section's text (heading None), then
    one part per other section it runs on into. A line starts another section when it is exactly
    the name of a section in `paths` that is not inside the passage's own section (its own
    sub-sections, like a release's "Fixed", don't end it)."""
    own_parts = {p[len(section) + 3:] for p in paths if p.startswith(section + " > ")}
    inside = {name for p in own_parts for name in p.split(" > ")}
    others = {name for p in paths for name in p.split(" > ")} - inside
    first, _, rest = passage.partition("\n")
    parts, heading, lines = [], None, [first]
    for line in rest.split("\n"):
        if line.strip() in others:
            parts.append((heading, "\n".join(lines)))
            heading, lines = line.strip(), [line]
        else:
            lines.append(line)
    parts.append((heading, "\n".join(lines)))
    return parts


def same_heading(a: str | None, b: str | None) -> bool:
    a, b = (re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip() for x in (a, b))
    return bool(a and b) and (a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a)))


def _versions(text: str) -> set[str]:
    """Version-like codes (0.14.5, v2.11.8), not dates or ordinals that many sections share."""
    return {code.lstrip("v") for code in identifiers(text) if re.fullmatch(r"v?\d+(?:\.\d+)+[a-z0-9]*", code)}


def _belongs(segment_heading: str | None, claim: str, under: str | None) -> bool:
    """May a claim use text a passage runs on into from this other section? Yes when the claim is
    written under that section's heading, or names its version ("v2.11.8: Fix mypy plugin")."""
    return same_heading(segment_heading, under) or bool(_versions(segment_heading or "") & _versions(claim))


def check_citations(answer: str, passages: list[str], sections: list[str] | None = None,
                    paths: set[str] | frozenset = frozenset()) -> list[CitationCheck]:
    """Check every cited sentence of `answer` against `passages` (passage id i is passages[i-1]).
    With `sections` (each passage's section path) and `paths` (every section path in their
    documents), a passage supports a claim only with its own section's text, plus any text it runs
    on into from the section the claim is written under. Without them, the whole passage counts."""
    split = [segments(p, s, paths) if sections and paths else [(None, p)]
             for p, s in zip(passages, sections or passages, strict=True)]
    split_words = [[(h, content_words(t)) for h, t in parts] for parts in split]
    results = []
    for claim, cited, under in claims(answer):
        words = content_words(claim)
        if len(words) < MIN_WORDS:
            continue
        passage_words = [set().union(*(w for h, w in parts if h is None or under is None or _belongs(h, claim, under)))
                         for parts in split_words]
        scores = [overlap(words, pw) for pw in passage_words]
        valid = [i for i in cited if 1 <= i <= len(passages)]
        score = max((scores[i - 1] for i in valid), default=0.0)
        best = max(range(len(scores)), key=scores.__getitem__) + 1 if scores else None
        best_score = scores[best - 1] if best else 0.0
        if score >= SUPPORTED:
            status, best = "supported", None
        elif best_score >= SUPPORTED and best_score >= score + MISCITED_MARGIN:
            status = "miscited"
        else:
            status, best = "unsupported", None
        results.append(CitationCheck(claim, cited, status, round(score, 2), best, round(best_score, 2)))
    return results
