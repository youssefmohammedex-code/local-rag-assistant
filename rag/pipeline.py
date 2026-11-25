"""Retrieve-then-generate pipeline."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from dataclasses import dataclass

import ollama

from rag.chunking import chunk_pages
from rag.config import Settings
from rag.errors import NoDocumentsError
from rag.loaders import load_documents, pretty_source
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


def candidate_lists(hits: list[Hit], outlines: dict[str, list[tuple[str, str]]]) -> dict[tuple[str, str], list]:
    """Every section that could be "the list": each candidate's own section, all its ancestors, and
    the file's top level (for documents without a title heading). Only sections with 2+ sub-sections."""
    nodes: dict[tuple[str, str], list] = {}
    for hit in hits:
        parts = hit.section.split(" > ") if hit.section else []
        for depth in range(len(parts) + 1):
            key = (hit.source, " > ".join(parts[:depth]))
            if key not in nodes:
                kids = children_in_outline(outlines[hit.source], key[1])
                nodes[key] = kids if len(kids) >= 2 else []
    return {key: kids for key, kids in nodes.items() if kids}


def describe_list(source: str, parent: str, kids: list) -> str:
    """A section summarized as its full heading path plus its sub-headings, e.g.
    "orjson > Types: dataclass; datetime; enum; ...". The full path matters: with only the last
    heading, the whole-document summary was the only one containing the product name, so any
    question naming the product ("all the types orjson can serialize") picked the whole document."""
    path = parent or pretty_source(source)
    return f"{path}: " + "; ".join(name for name, _ in kids[:MAX_NODE_CHILDREN])


CITATION_RULE ='After each fact, put the passage id in brackets, for example: "... a GPA of 3.0 [2]."'


def build_prompt(question: str, hits: list[Hit], mode: str = "detailed") -> str:
    # XML-style tags keep small models from copying passage headers into the answer.
    blocks = []
    for i, h in enumerate(hits, start=1):
        attrs = f'id="{i}" source="{pretty_source(h.source)}"'
        attrs += f' section="{h.section}"' if h.section else f' page="{h.page}"'
        blocks.append(f"<passage {attrs}>\n{h.text}\n</passage>")
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    return f"{context}\n\nQuestion: {question}\n\n{MODES[mode].instructions} {CITATION_RULE}"


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
        self.store = VectorStore(self.settings)
        self._llm = ollama.Client(host=self.settings.ollama_host)
        self.last_stats = GenerationStats()
        self.list_scorer = self.store.similarities  # (question, texts) -> scores; replaceable in tests

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

    def pick_list(self, question: str, candidates: list[Hit]) -> tuple[str, str, list] | None:
        """Which section is the question asking to enumerate? Each possible section is summarized as
        "heading: sub-heading; sub-heading; ..." and the one whose summary is most similar to the
        question (same embedding model as search) wins. No hand-tuned weights: an earlier
        weighted-heuristic version overfit one document set and failed on new documents."""
        outlines = {source: self.store.outline(source) for source in {h.source for h in candidates}}
        nodes = candidate_lists(candidates, outlines)
        if not nodes:
            return None
        keys = list(nodes)
        scores = self.list_scorer(question, [describe_list(s, p, nodes[(s, p)]) for s, p in keys])
        source, parent = keys[max(range(len(keys)), key=scores.__getitem__)]
        return source, parent, nodes[(source, parent)]

    def _retrieve_list(self, question: str, k: int, mode: str) -> list[Hit]:
        """List questions ("describe every module") need one passage per item, not the k most
        similar passages (which cluster on a few items). Pick the section being enumerated, keep
        the best few normal hits, then add the first passage of *every* sub-section, trimming
        each one so the whole list fits the context budget."""
        candidates = self.store.search(question, max(k, LIST_CANDIDATES), mode)
        picked = self.pick_list(question, candidates)
        if picked is None:
            return candidates[:k]
        _, _, kids = picked
        # Additive by design: keep exactly the normal top-k and only add list items in the space
        # left. Picking the right section is ambiguous and every selection rule tried failed on
        # some documents; this way a wrong pick costs context space, never the normal results.
        top = candidates[:k]
        top_ids = {t.id for t in top}
        items = self.store.hits_for([cid for _, cid in kids if cid not in top_ids], question, via="section")
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

    def _request(self, question: str, hits: list[Hit], mode: str | None) -> dict:
        return {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits, mode or self.settings.answer_mode)},
            ],
            "options": {"temperature": self.settings.temperature, "num_ctx": self.settings.num_ctx},
        }

    def generate(self, question: str, hits: list[Hit], mode: str | None = None) -> str:
        response = self._llm.chat(**self._request(question, hits, mode))
        return response["message"]["content"].strip()

    def stream(self, question: str, hits: list[Hit], mode: str | None = None) -> Iterator[str]:
        self.last_stats = GenerationStats()
        for part in self._llm.chat(**self._request(question, hits, mode), stream=True):
            if part.get("done"):
                self.last_stats = GenerationStats(
                    tokens=part.get("eval_count") or 0,
                    seconds=(part.get("eval_duration") or 0) / 1e9,
                )
            yield part["message"]["content"]

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
