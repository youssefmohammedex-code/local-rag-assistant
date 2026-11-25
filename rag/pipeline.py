"""Retrieve-then-generate pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import ollama

from rag.chunking import chunk_pages
from rag.config import Settings
from rag.loaders import load_documents
from rag.store import Hit, VectorStore

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you use inline as [1], [2], etc. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\" Be concise."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    # XML-style tags keep small models from copying passage headers into the answer.
    blocks = [
        f'<passage id="{i}" source="{h.source}" page="{h.page}">\n{h.text}\n</passage>'
        for i, h in enumerate(hits, start=1)
    ]
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    return (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Write a complete answer in 1-3 sentences using only the passages above. "
        'After each fact, put the passage id in brackets, for example: "... a GPA of 3.0 [2]."'
    )


@dataclass
class Answer:
    text: str
    hits: list[Hit]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.store = VectorStore(self.settings)
        self._llm = ollama.Client(host=self.settings.ollama_host)

    def ingest(self, reset: bool = True) -> int:
        """Load, chunk and embed every document. Returns the number of chunks stored."""
        pages = load_documents(self.settings.docs_dir)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if reset:
            self.store.reset()
        self.store.add(chunks)
        return len(chunks)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.search(question, k or self.settings.top_k)

    def _messages(self, question: str, hits: list[Hit]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ]

    def generate(self, question: str, hits: list[Hit]) -> str:
        response = self._llm.chat(
            model=self.settings.llm_model,
            messages=self._messages(question, hits),
            options={"temperature": self.settings.temperature},
        )
        return response["message"]["content"].strip()

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        for part in self._llm.chat(
            model=self.settings.llm_model,
            messages=self._messages(question, hits),
            options={"temperature": self.settings.temperature},
            stream=True,
        ):
            yield part["message"]["content"]

    def answer(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k)
        return Answer(self.generate(question, hits), hits)
