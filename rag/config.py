"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DOCS_DIR", ROOT / "data" / "docs")))
    db_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DB_DIR", ROOT / "data" / "chroma")))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("RAG_EMBED_MODEL", "nomic-embed-text"))
    llm_model: str = field(default_factory=lambda: os.getenv("RAG_LLM_MODEL", "llama3.2:3b"))
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 4))
    temperature: float = 0.1

    @property
    def collection_name(self) -> str:
        """One collection per chunking config, so different configs can be compared side by side."""
        return f"docs_cs{self.chunk_size}_ov{self.chunk_overlap}"
