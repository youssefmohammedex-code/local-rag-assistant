"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Bump when chunking/loading changes so old indexes are not silently reused.
INDEX_VERSION = 2


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
    # Ollama's default context window is small; 10 passages + a long answer needs more room.
    num_ctx: int = field(default_factory=lambda: _env_int("RAG_NUM_CTX", 8192))
    temperature: float = 0.1

    @property
    def collection_name(self) -> str:
        """One collection per chunking config, so different configs can be compared side by side."""
        return f"docs_v{INDEX_VERSION}_cs{self.chunk_size}_ov{self.chunk_overlap}"
