"""Retrieve-then-generate pipeline."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from dataclasses import dataclass

import ollama

from rag.bm25 import identifiers
from rag.chunking import chunk_pages
from rag.citations import CitationCheck, check_citations, same_heading, segments
from rag.config import Settings
from rag.errors import NoDocumentsError
from rag.filler import filler_headings, is_filler, splice
from rag.filler import sections as answer_sections
from rag.loaders import load_documents, pretty_source
from rag.loops import LoopDetector
from rag.store import Hit, VectorStore, children_in_outline

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions about the user's documents. "
    "Use ONLY the numbered passages you are given; never invent facts. "
    "If the passages do not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\" "
    "If they only partly answer it, answer that part and say what is missing."
)


@dataclass(frozen=True)
class Mode:
    k: int
    instructions: str


MODES: dict[str, Mode] = {
    "detailed": Mode(
        k=10,
        instructions=(
            "Write a thorough, complete answer using every relevant passage above. "
            "If the question asks about several things (for example 'all', 'every', 'list', "
            "'compare', 'explain'), cover each one; do not stop after the first few. "
            "Format with Markdown: short headings or bullet points for multi-part answers, "
            "and keep important specifics such as names, numbers, rules and steps."
        ),
    ),
    "concise": Mode(k=4, instructions="Answer in 1-3 sentences."),
}

_LIST_QUESTION = re.compile(r"\b(all|every|each|list|enumerate)\b", re.IGNORECASE)
LIST_CANDIDATES = 20  # candidates whose sections (and their ancestors) are considered as "the list"
MIN_ITEM_CHARS = 350  # never trim a list item below this (heading + first lines)
MAX_NODE_CHILDREN = 60  # sub-headings included when describing a section for selection


def is_list_question(question: str) -> bool:
    return bool(_LIST_QUESTION.search(question))


def names_code(name: str, code: str) -> bool:
    """True if a heading contains an exact code from the question as a whole code or a prefix of one:
    "2.14" matches "v2.14.4 (2026-10-08)" but not "v12.14" or "2.140"; "v2.14.0" does not match "v2.14.0b1"."""
    return re.search(rf"(?<![0-9.]){re.escape(code)}(?![0-9a-z])", name.lower()) is not None


def filter_by_codes(kids: list, codes: set[str]) -> list:
    """Sub-sections named by a code in the question ("every 2.14 release" -> v2.14.0, v2.14.1, ...).
    Unchanged unless at least 2 match: the question then asks for that subset, not the whole list
    (a changelog with 100+ releases summarizes as a wall of version numbers and never ranks first)."""
    matching = [kid for kid in kids if any(names_code(kid[0], code) for code in codes)]
    return matching if len(matching) >= 2 else kids


def candidate_lists(hits: list[Hit], outlines: dict[str, list[tuple[str, str]]],
                    codes: set[str] | None = None) -> dict[tuple[str, str], list]:
    """Every section that could be "the list": each candidate's own section, all its ancestors, and
    the file's top level (for documents without a title heading). Only sections with 2+ sub-sections.
    With codes from the question, a section's sub-sections are narrowed to those the codes name."""
    nodes: dict[tuple[str, str], list] = {}
    for hit in hits:
        parts = hit.section.split(" > ") if hit.section else []
        for depth in range(len(parts) + 1):
            key = (hit.source, " > ".join(parts[:depth]))
            if key not in nodes:
                kids = children_in_outline(outlines[hit.source], key[1])
                if codes:
                    kids = filter_by_codes(kids, codes)
                nodes[key] = kids if len(kids) >= 2 else []
    return {key: kids for key, kids in nodes.items() if kids}


# Which ranked sections to expand. Benchmarked on 29 list questions over 4 document sets:
# top1 86%, ancestor 84%, nested-both 94%, top2 97% coverage, and top2 made no question worse than
# top1. The selector is often unsure (0.793 vs 0.791 between a changelog and one release's
# sub-list), so expanding the two best hedges that, at ~27% more context. Kept switchable.
LIST_STRATEGY = "top2"  # "top1" | "top2" | "nested-both" | "ancestor"


def _is_ancestor(a: tuple, b: tuple) -> bool:
    """True if section a contains section b (same file)."""
    return a[0] == b[0] and a[1] != b[1] and (a[1] == "" or b[1].startswith(a[1] + " > "))


def _nested(a: tuple, b: tuple) -> bool:
    return _is_ancestor(a, b) or _is_ancestor(b, a)


def list_heading(source: str, parent: str, kids: list) -> str:
    """The heading a candidate list is named by: its section path, or the file for a top-level list."""
    return parent or pretty_source(source)


def describe_list(source: str, parent: str, kids: list) -> str:
    """A section summarized as its full heading path plus its sub-headings, e.g.
    "orjson > Types: dataclass; datetime; enum; ...". The full path matters: with only the last
    heading, the whole-document summary was the only one containing the product name, so any
    question naming the product ("all the types orjson can serialize") picked the whole document."""
    return f"{list_heading(source, parent, kids)}: " + "; ".join(name for name, _ in kids[:MAX_NODE_CHILDREN])


def heading_matches(reply: str, headings: list[str]) -> int | None:
    """Which heading a reply names: the one it equals in full, else the one whose last part it
    equals ("Other Use Cases" for "python-dotenv > Other Use Cases"). None if not exactly one."""
    def norm(text: str) -> str:
        return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())
    said = norm(reply.strip().strip("\"'`*.").split("\n")[0])
    for parts in (lambda h: h, lambda h: h.split(" > ")[-1]):
        found = [i for i, h in enumerate(headings) if norm(parts(h)) == said]
        if len(found) == 1:
            return found[0]
    return None


CITATION_RULE ='After each fact, put the passage id in brackets, for example: "... a GPA of 3.0 [2]."'


def build_prompt(question: str, hits: list[Hit], mode: str = "detailed", ids: list[int] | None = None,
                 instructions: str | None = None) -> str:
    # XML-style tags keep small models from copying passage headers into the answer.
    blocks = []
    for i, h in zip(ids or range(1, len(hits) + 1), hits, strict=True):
        attrs = f'id="{i}" source="{pretty_source(h.source)}"'
        if h.section:
            attrs += f' section="{h.section}"'
        if not h.section or h.source.lower().endswith(".pdf"):  # PDFs: page numbers matter for citations
            attrs += f' page="{h.page}"'
        blocks.append(f"<passage {attrs}>\n{h.text}\n</passage>")
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    return f"{context}\n\nQuestion: {question}\n\n{instructions or MODES[mode].instructions} {CITATION_RULE}"


@dataclass
class ListItem:
    """One item of a list answer: a sub-section and the retrieved passages (1-based ids) about it."""
    name: str
    path: str
    passages: list[int]


LIST_ANSWERS = ("checklist", "single", "per-item")  # see RAGPipeline.stream
LIST_ITEMS_PER_CALL = 3  # per-item answers: sections per model call
LOOP_NOTE = "\n\n*(Answer stopped early: the model started repeating itself.)*"


def checklist_instructions(items: list[ListItem]) -> str:
    """Show the model the outline of the expanded list(s) so it cannot silently skip items.
    Deliberately soft: retrieval expands two candidate lists and one is often unrelated, and a
    strict "exactly these headings" version forced that list into the answer (orjson FAQ 100% -> 80%).
    Tried and reverted: listing each section's passage ids. It did not stop invented content (httpcore
    guide) and lowered coverage on 3 of 13 questions."""
    headings = "\n".join(f"### {item.name}" for item in items)
    return ("The passages come from these sections of the documents, in this order:\n"
            f"{headings}\n"
            "Use the sections that answer the question as headings, in this order, and do not skip any "
            "of them: go through the list to the end, and when unsure whether a section is relevant, "
            "include it. Leave out only sections clearly unrelated to the question. If the items the "
            "question asks for are listed inside a section's text rather than as sections, use those "
            "items as headings instead. Under each heading write 1-3 bullet points using only the "
            "passages from that section. Do not move examples or facts between headings.")


def _document_order(hit: Hit, hits: list[Hit]) -> tuple:
    """Sort key: documents in order of their first appearance in `hits`, then position in the
    document (chunk ids look like "<source>::p<page>::s<part>::c<i>")."""
    match = re.search(r"::s(\d+)::c(\d+)$", hit.id)
    position = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    first = next(i for i, h in enumerate(hits) if h.source == hit.source)
    return (first, *position)


def choose_list_prompt(question: str, lists: list) -> str:
    options = "\n".join(f"{letter}: {describe_list(*lst)}" for letter, lst in zip("AB", lists, strict=True))
    return (f"Question: {question}\n\nTwo lists of sections from the documents:\n{options}\n\n"
            "Which list's sections does the question ask about? Answer with the single letter A or B.")


def choose_heading_prompt(question: str, lists: list) -> str:
    """The same choice, answered with the list's heading instead of a letter (see choose_list)."""
    options = "\n".join(f"- {describe_list(*lst)}" for lst in lists)
    return (f"Question: {question}\n\nTwo lists of sections from the documents, each written as "
            f"'heading: its sections':\n{options}\n\nWhich list's sections does the question ask about? "
            "Answer with that list's heading only, exactly as written before the colon.")


REPAIR_QUESTION = "What does each of these sections of the documents say?"
REPAIR_ITEMS_PER_CALL = 8  # filler sections regenerated per model call


def plain_name(name: str) -> str:
    """A section name without square brackets: Keep-a-Changelog headings like "[0.17.1] - 2021-04-29"
    made the model write passage ids ("### [26]") as headings."""
    return re.sub(r"\s+", " ", name.replace("[", "").replace("]", "")).strip()


def repair_instructions(items: list[ListItem]) -> str:
    names = "\n".join(f"### {plain_name(item.name)}" for item in items)
    return ("For each of these sections, in this order, write its heading exactly as given and under it "
            "1-3 bullet points with the main points of the passages from that section:\n"
            f"{names}\nUse only the passages from that section. Do not add any other headings or an introduction.")


def fix_item_citations(body: str, passages: list[int]) -> str:
    """Citations in a regenerated section must be that section's passages. Others (often a pull
    request number like [#700] copied as [700]) become its passage if it has one, else are dropped.
    A section regenerated from one passage can only say what that passage says, so its uncited
    bullets get that citation and a lone "[34]" line after them is removed (the model left 213 of
    378 bullets uncited)."""
    def fix(match: re.Match) -> str:
        ids = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if all(i in passages for i in ids):
            return match.group(0)
        return f"[{passages[0]}]" if len(passages) == 1 else ""
    body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", fix, body)
    if len(passages) != 1:
        return body
    lines = [line for line in body.split("\n") if not re.fullmatch(r"\s*\[\d+\]\s*", line)]
    cite = f"[{passages[0]}]"
    return "\n".join(line if not re.match(r"\s*[-*] ", line) or re.search(r"\[\d+\]", line)
                     else f"{line.rstrip()} {cite}" for line in lines)


def per_item_instructions(items: list[ListItem]) -> str:
    names = "; ".join(item.name for item in items)
    return (f"Answer the question separately for each of these sections: {names}. "
            "For each one, write a \"### <section name>\" heading followed by 1-3 bullet points, using only "
            "the passages about that section. Do not add any other headings or an introduction.")


@dataclass
class Answer:
    text: str
    hits: list[Hit]


@dataclass
class IngestReport:
    chunks: int
    files: int
    skipped: list[str]  # files that produced no text (e.g. scanned PDFs without a text layer)
    unsupported: list[str] = dataclasses.field(default_factory=list)  # e.g. .docx, .html


@dataclass
class GenerationStats:
    tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        if self.settings.answer_mode not in MODES:
            raise ValueError(f"answer_mode must be one of {sorted(MODES)}")
        if self.settings.list_answers not in LIST_ANSWERS:
            raise ValueError(f"list_answers must be one of {list(LIST_ANSWERS)}")
        self.store = VectorStore(self.settings)
        self._llm = ollama.Client(host=self.settings.ollama_host)
        self.last_stats = GenerationStats()
        self.list_scorer = self.store.similarities  # (question, texts) -> scores; replaceable in tests
        self._list_plans: dict[str, list[tuple[str, str, list]]] = {}  # question -> lists picked at retrieval

    def ingest(self, reset: bool = True) -> IngestReport:
        """Load, chunk and embed every document, then replace the index.

        Safe on failure: all embeddings are computed *before* the old index is touched, so if
        Ollama is down (or anything else fails while embedding) the existing index is kept.
        An empty or misconfigured docs folder raises NoDocumentsError instead of wiping the index.
        """
        skipped: list[str] = []
        unsupported: list[str] = []
        pages = load_documents(self.settings.docs_dir, skipped=skipped, unsupported=unsupported)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise NoDocumentsError(
                f"No readable documents found in {self.settings.docs_dir} (supported: .pdf, .md, .txt). "
                "The existing index was left unchanged."
            )
        embeddings = self.store.embed_chunks(chunks)
        if reset:
            self.store.reset()
        self.store.add(chunks, embeddings)
        return IngestReport(chunks=len(chunks), files=len({c.source for c in chunks}), skipped=skipped,
                            unsupported=unsupported)

    def default_k(self, mode: str | None = None) -> int:
        return self.settings.top_k or MODES[mode or self.settings.answer_mode].k

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        k, mode = k or self.default_k(), mode or self.settings.retrieval
        if not (self.settings.expand_lists and is_list_question(question)):
            return self.store.search(question, k, mode)
        return self._retrieve_list(question, k, mode)

    def rank_lists(self, question: str, candidates: list[Hit]) -> list[tuple[str, str, list]]:
        """Possible sections to enumerate, best first. Each is summarized as "heading path:
        sub-heading; sub-heading; ..." and ranked by similarity to the question (same embedding
        model as search). No hand-tuned weights: an earlier weighted-heuristic version overfit one
        document set and failed on new documents."""
        outlines = {source: self.store.outline(source) for source in {h.source for h in candidates}}
        nodes = candidate_lists(candidates, outlines, identifiers(question))
        if not nodes:
            return []
        keys = list(nodes)
        scores = self.list_scorer(question, [describe_list(s, p, nodes[(s, p)]) for s, p in keys])
        order = sorted(range(len(keys)), key=lambda i: -scores[i])
        return [(*keys[i], nodes[keys[i]]) for i in order]

    def pick_lists(self, question: str, candidates: list[Hit]) -> list[tuple[str, str, list]]:
        ranked = self.rank_lists(question, candidates)
        if not ranked:
            return []
        if LIST_STRATEGY == "top2":
            return ranked[:2]
        if LIST_STRATEGY == "nested-both" and len(ranked) > 1 and _nested(ranked[0], ranked[1]):
            return ranked[:2]
        if LIST_STRATEGY == "ancestor":
            for other in ranked[1:3]:
                if _is_ancestor(other, ranked[0]):
                    return [other]
        return ranked[:1]

    def _retrieve_list(self, question: str, k: int, mode: str) -> list[Hit]:
        """List questions ("describe every module") need one passage per item, not the k most
        similar passages (which cluster on a few items). Pick the section being enumerated, keep
        the best few normal hits, then add the first passage of *every* sub-section, trimming
        each one so the whole list fits the context budget."""
        candidates = self.store.search(question, max(k, LIST_CANDIDATES), mode)
        picked = self.pick_lists(question, candidates)
        if not picked:
            return candidates[:k]
        if len(self._list_plans) > 100:
            self._list_plans.clear()
        self._list_plans[question] = picked
        # A parent's sub-section and its child's first item can start with the same passage: dedupe by id.
        chunk_ids = list(dict.fromkeys(cid for _, _, node_kids in picked for _, cid in node_kids))
        # Additive by design: keep exactly the normal top-k and only add list items in the space
        # left. Picking the right section is ambiguous and every selection rule tried failed on
        # some documents; this way a wrong pick costs context space, never the normal results.
        top = candidates[:k]
        top_ids = {t.id for t in top}
        items = self.store.hits_for([cid for cid in chunk_ids if cid not in top_ids], question, via="section")
        budget = self.settings.list_char_budget - sum(len(h.text) for h in top)
        per_item = max(MIN_ITEM_CHARS, budget // max(len(items), 1))
        expanded, used = [], 0
        for item in items:
            item = item.truncated(per_item)
            if used + len(item.text) > budget:
                break  # still too many items: keep the first ones in document order
            expanded.append(item)
            used += len(item.text)
        return top + expanded

    def list_items(self, question: str, hits: list[Hit], lists: list | None = None) -> list[ListItem]:
        """The list a question was expanded into, as items with the passages about each one.
        Empty when retrieval did not treat it as a list question."""
        items: dict[str, ListItem] = {}
        for source, parent, kids in self._list_plans.get(question, []) if lists is None else lists:
            for name, _ in kids:
                path = f"{parent} > {name}" if parent else name
                if path in items:
                    continue
                passages = [i for i, h in enumerate(hits, start=1)
                            if h.source == source and (h.section == path or h.section.startswith(path + " > "))]
                if passages:
                    items[path] = ListItem(name, path, passages)
        return list(items.values())

    def _options(self, **options) -> dict:
        """Ollama options for one call: context size, the given ones, and the seed when one is set."""
        options = {"num_ctx": self.settings.num_ctx, **options}
        if self.settings.seed is not None:
            options["seed"] = self.settings.seed
        return options

    def _request(self, question: str, hits: list[Hit], mode: str | None, ids: list[int] | None = None,
                 instructions: str | None = None) -> dict:
        prompt = build_prompt(question, hits, mode or self.settings.answer_mode, ids=ids, instructions=instructions)
        return {
            "model": self.settings.llm_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "options": self._options(temperature=self.settings.temperature,
                                     num_predict=self.settings.max_answer_tokens),
        }

    def choose_list(self, question: str) -> int | None:
        """Retrieval expands the two best-matching lists because embedding similarity often can't
        tell them apart (it ranked an httpcore changelog above the guide's own sections for "all
        the sections of the guide"). Before answering, ask the model which one the question means:
        a one-letter multiple choice that a 7B model gets right far more reliably than it ignores
        40 irrelevant passages. Asked in both orders, because the model leans towards one letter
        when unsure: a pick counts only if both orders agree.

        When the letters disagree, the model was answering by position ("B" both times, for Typer's
        example upgrade and python-dotenv's use cases), so it is asked once more per order to name the
        list's heading instead, which it can't do by position. Only these abstentions reach the second
        question, so every pick the letters already agree on is unchanged. None (keep both lists, as
        before) when there is nothing to choose or neither question gets the same answer in both orders."""
        lists = self._list_plans.get(question, [])
        if len(lists) != 2:
            return None
        letters = [self._ask_choice(question, lists, order, by_heading=False) for order in ([0, 1], [1, 0])]
        if letters[0] is not None and letters[0] == letters[1]:
            return letters[0]
        headings = [self._ask_choice(question, lists, order, by_heading=True) for order in ([0, 1], [1, 0])]
        return headings[0] if headings[0] is not None and headings[0] == headings[1] else None

    def _ask_choice(self, question: str, lists: list, order: list[int], by_heading: bool) -> int | None:
        """One chooser question with the lists in `order`; the index of the list picked, or None."""
        shown = [lists[i] for i in order]
        prompt = choose_heading_prompt(question, shown) if by_heading else choose_list_prompt(question, shown)
        request = {"model": self.settings.llm_model, "messages": [{"role": "user", "content": prompt}],
                   "options": self._options(temperature=0, num_predict=24 if by_heading else 4)}
        reply = "".join(self._stream_call(request)).strip()
        if not by_heading:
            match = re.match(r"\W*([AB])\b", reply.upper())
            return order["AB".index(match.group(1))] if match else None
        picked = heading_matches(reply, [list_heading(*lst) for lst in shown])
        return order[picked] if picked is not None else None

    def generate(self, question: str, hits: list[Hit], mode: str | None = None) -> str:
        """The full answer, with filler sections of a list answer regenerated (repair_filler).
        Streaming callers (app, CLI) call repair_filler themselves after the stream ends."""
        answer = "".join(self.stream(question, hits, mode)).strip()
        return self.repair_filler(question, hits, answer)[0]

    def _stream_call(self, request: dict) -> Iterator[str]:
        """Stream one model call. Stops early, with a note, if the model starts repeating itself
        (see rag.loops); closing the stream makes Ollama cancel the generation."""
        if self.settings.cold_calls:
            request = {**request, "keep_alive": 0}
        stream = self._llm.chat(**request, stream=True)
        detector = LoopDetector()
        try:
            for part in stream:
                if part.get("done"):
                    self.last_stats = GenerationStats(
                        tokens=self.last_stats.tokens + (part.get("eval_count") or 0),
                        seconds=self.last_stats.seconds + (part.get("eval_duration") or 0) / 1e9,
                    )
                text = part["message"]["content"]
                yield text
                if detector.feed(text):
                    yield LOOP_NOTE
                    return
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

    def stream(self, question: str, hits: list[Hit], mode: str | None = None) -> Iterator[str]:
        """Stream an answer. Detailed answers to list questions use the configured list strategy
        (settings.list_answers): "single" one call, "checklist" one call that must use exactly the
        list's headings, or "per-item" one call per few items with only those items' passages."""
        self.last_stats = GenerationStats()
        mode = mode or self.settings.answer_mode
        strategy = self.settings.list_answers
        items = self.list_items(question, hits) if mode == "detailed" and strategy != "single" else []
        if not items:
            yield from self._stream_call(self._request(question, hits, mode))
            return
        if strategy == "checklist":
            ids = list(range(1, len(hits) + 1))
            choice = self.choose_list(question)
            if choice is not None:
                # Answer from the chosen list only: keep the normal top-k (as always) plus the passages
                # expanded for that list, and drop the other list's.
                chosen = self._list_plans[question][choice]
                expanded_for_it = {chunk_id for _, chunk_id in chosen[2]}
                ids = [i for i, h in enumerate(hits, start=1) if h.via != "section" or h.id in expanded_for_it]
                kept = set(ids)
                items = [dataclasses.replace(item, passages=[i for i in item.passages if i in kept])
                         for item in self.list_items(question, hits, lists=[chosen])]
                items = [item for item in items if item.passages] or self.list_items(question, hits)
            # Show passages in document order (the outline's order), keeping their real ids, so each
            # section's passages sit together under the heading order the model is asked to follow.
            ids = sorted(ids, key=lambda i: _document_order(hits[i - 1], hits))
            request = self._request(question, [hits[i - 1] for i in ids], mode, ids=ids,
                                    instructions=checklist_instructions(items))
            yield from self._stream_call(request)
            return
        for start in range(0, len(items), LIST_ITEMS_PER_CALL):
            batch = items[start:start + LIST_ITEMS_PER_CALL]
            ids = sorted({i for item in batch for i in item.passages})
            request = self._request(question, [hits[i - 1] for i in ids], mode, ids=ids,
                                    instructions=per_item_instructions(batch))
            yield from self._stream_call(request)
            yield "\n\n"

    def repair_filler(self, question: str, hits: list[Hit], answer: str) -> tuple[str, int]:
        """Regenerate the sections of a list answer that are only filler (see rag.filler), without
        the question that triggered it: a few sections per call, each call seeing only those
        sections' passages under their real ids, cut to the section's own text (a passage runs on
        into the next sections, and the model mixed a neighbouring release's changes in). Returns
        (answer, sections repaired). A section is kept as it was if it can't be matched to a list
        item or its new text is filler again."""
        filler = filler_headings(answer)
        if not filler:
            return answer, 0
        items = self.list_items(question, hits)
        todo = []
        for heading in filler:
            item = next((i for i in items if same_heading(i.name, heading)), None)
            if item and all(item is not other for _, other in todo):
                todo.append((heading, item))
        if not todo:
            return answer, 0
        paths = self.section_paths(hits)
        bodies: dict[str, str] = {}
        for start in range(0, len(todo), REPAIR_ITEMS_PER_CALL):
            batch = todo[start:start + REPAIR_ITEMS_PER_CALL]
            ids = sorted({i for _, item in batch for i in item.passages})
            shown = [dataclasses.replace(hits[i - 1], text=segments(hits[i - 1].text, hits[i - 1].section, paths)[0][1])
                     for i in ids]
            request = self._request(REPAIR_QUESTION, shown, "detailed", ids=ids,
                                    instructions=repair_instructions([item for _, item in batch]))
            text = "".join(self._stream_call(request))
            written = [(h, text[s:e]) for h, s, e in answer_sections(text)]
            for heading, item in batch:
                body = next((b for h, b in written if same_heading(h, plain_name(item.name))), None)
                if body is None:  # "### [26]": the model wrote the section's passage id as its heading
                    body = next((b for h, b in written if re.fullmatch(r"\[?(\d+)\]?", h.strip())
                                 and int(h.strip("[] ")) in item.passages), None)
                if body is not None and not is_filler(body):
                    bodies[heading] = fix_item_citations(body, item.passages)
        return splice(answer, bodies), len(bodies)

    def section_paths(self, hits: list[Hit]) -> set[str]:
        """Every section path in the documents the hits come from."""
        return {section for source in {h.source for h in hits}
                for _, section in self.store.outline(source) if section}

    def check_citations(self, answer: str, hits: list[Hit]) -> list[CitationCheck]:
        """Verify an answer's citations against the passages it was given (see rag.citations)."""
        return check_citations(answer, [h.text for h in hits], [h.section for h in hits], self.section_paths(hits))

    def answer(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        hits = self.retrieve(question, k or self.default_k(mode))
        return Answer(self.generate(question, hits, mode), hits)

    def installed_models(self) -> list[str]:
        """Chat models available in the local Ollama install (embedding models excluded)."""
        try:
            names = [m["model"] for m in self._llm.list()["models"]]
        except Exception:
            return []
        return sorted(n for n in names if "embed" not in n)
