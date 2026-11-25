"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Bump when chunking/loading changes so old indexes are not silently reused.
INDEX_VERSION = 3  # v3: PDF headings/sections, context for tiny sections, stemmed keywords


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DOCS_DIR", ROOT / "data" / "docs")))
    db_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DB_DIR", ROOT / "data" / "chroma")))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("RAG_EMBED_MODEL", "nomic-embed-text"))
    llm_model: str = field(default_factory=lambda: os.getenv("RAG_LLM_MODEL", "qwen2.5:7b"))
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 150))
    # "detailed" or "concise". Each mode has its own default number of passages (see pipeline.MODES).
    answer_mode: str = field(default_factory=lambda: os.getenv("RAG_ANSWER_MODE", "detailed"))
    top_k: int | None = field(default_factory=lambda: _env_int("RAG_TOP_K", None))
    # "hybrid" (embeddings + BM25 keyword search) or "vector" (embeddings only).
    retrieval: str = field(default_factory=lambda: os.getenv("RAG_RETRIEVAL", "hybrid"))
    # List questions ("describe every module") get one extra passage per sub-section on top of the
    # normal top-k, within a total budget (~20k characters, about 5k tokens) that leaves room in
    # num_ctx for a long answer.
    expand_lists: bool = field(default_factory=lambda: os.getenv("RAG_EXPAND_LISTS", "1") != "0")
    list_char_budget: int = field(default_factory=lambda: _env_int("RAG_LIST_CHAR_BUDGET", 20000))
    # How detailed answers to list questions are generated: "checklist" (the model is shown the
    # outline of the expanded sections so it cannot silently skip one), "single" or "per-item".
    list_answers: str = field(default_factory=lambda: os.getenv("RAG_LIST_ANSWERS", "checklist"))
    # Ollama's default context window is small; 10 passages + a long answer needs more room.
    num_ctx: int = field(default_factory=lambda: _env_int("RAG_NUM_CTX", 8192))
    temperature: float = 0.1

    @property
    def collection_name(self) -> str:
        """One collection per chunking config, so different configs can be compared side by side."""
        return f"docs_v{INDEX_VERSION}_cs{self.chunk_size}_ov{self.chunk_overlap}"
