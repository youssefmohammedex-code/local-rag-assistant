"""Retrieve-then-generate pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import ollama

from rag.chunking import chunk_pages
from rag.config import Settings
from rag.loaders import load_documents, pretty_source
from rag.store import Hit, VectorStore

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

CITATION_RULE = 'After each fact, put the passage id in brackets, for example: "... a GPA of 3.0 [2]."'


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

    def ingest(self, reset: bool = True) -> int:
        """Load, chunk and embed every document. Returns the number of chunks stored."""
        pages = load_documents(self.settings.docs_dir)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if reset:
            self.store.reset()
        self.store.add(chunks)
        return len(chunks)

    def default_k(self, mode: str | None = None) -> int:
        return self.settings.top_k or MODES[mode or self.settings.answer_mode].k

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.search(question, k or self.default_k())

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
