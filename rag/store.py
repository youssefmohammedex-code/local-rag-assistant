"""Chroma vector store with embeddings computed locally by Ollama."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import chromadb
import ollama

from rag.chunking import Chunk
from rag.config import Settings


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better
    section: str = ""


class VectorStore:
    def __init__(self, settings: Settings, collection_name: str | None = None):
        self.settings = settings
        self.name = collection_name or settings.collection_name
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.db_dir))
        self._ollama = ollama.Client(host=settings.ollama_host)
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    # nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval.
    def _prefix(self, kind: str) -> str:
        if "nomic" not in self.settings.embed_model:
            return ""
        return "search_query: " if kind == "query" else "search_document: "

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        prefix = self._prefix(kind)
        response = self._ollama.embed(model=self.settings.embed_model, input=[prefix + t for t in texts])
        return response["embeddings"]

    def count(self) -> int:
        return self._collection.count()

    def sources(self) -> list[str]:
        """Distinct document paths currently in the index."""
        if self.count() == 0:
            return []
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metadatas})

    def reset(self) -> None:
        with contextlib.suppress(Exception):  # collection did not exist
            self._client.delete_collection(self.name)
        self._collection = self._get_collection()

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            vectors.extend(self.embed([c.text for c in chunks[i : i + batch_size]], kind="document"))
        return vectors

    def add(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None, batch_size: int = 256) -> None:
        """Store chunks. Pass precomputed `embeddings` to separate the slow, failure-prone
        embedding step from the write."""
        if embeddings is None:
            embeddings = self.embed_chunks(chunks)
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[i : i + batch_size],
                metadatas=[{"source": c.source, "page": c.page, "section": c.section} for c in batch],
            )

    def search(self, query: str, k: int) -> list[Hit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=self.embed([query], kind="query"),
            n_results=min(k, self.count()),
        )
        return [
            Hit(text=doc, source=meta["source"], page=int(meta["page"]), score=1.0 - dist,
                section=meta.get("section", ""))
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0],
                                       strict=True)
        ]
