"""Chroma vector store with embeddings computed locally by Ollama."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import chromadb
import numpy as np
import ollama

from rag.bm25 import BM25, is_identifier, reciprocal_rank_fusion, tokenize
from rag.chunking import Chunk
from rag.config import Settings

SEARCH_MODES = ("hybrid", "vector")
CANDIDATE_POOL = 50  # candidates taken from each ranking before fusion


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better
    section: str = ""
    via: str = "meaning"  # which search found it: "meaning" (vector), "keyword" (BM25) or "both"


class VectorStore:
    def __init__(self, settings: Settings, collection_name: str | None = None):
        self.settings = settings
        self.name = collection_name or settings.collection_name
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.db_dir))
        self._ollama = ollama.Client(host=settings.ollama_host)
        self._collection = self._get_collection()
        self._bm25: tuple[list[str], BM25] | None = None
        self._bm25_key: tuple | None = None

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
        self._bm25_key = None

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
        self._bm25_key = None

    def _db_stamp(self) -> tuple[int, ...]:
        """Changes whenever any process writes the index (Chroma keeps it in SQLite)."""
        files = [self.settings.db_dir / "chroma.sqlite3", self.settings.db_dir / "chroma.sqlite3-wal"]
        return tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)

    def _keyword_index(self) -> tuple[list[str], BM25]:
        """BM25 over every stored chunk, rebuilt only when the index changes."""
        key = (self.name, self.count(), self._db_stamp())
        if self._bm25_key != key:
            data = self._collection.get(include=["documents", "metadatas"])
            # Title boost: the heading path is already the chunk's first line; adding it once more
            # lets the section *named* "D13" outrank sections that only mention D13 in passing.
            corpus = [f"{m.get('section', '')}\n{d}" for d, m in zip(data["documents"], data["metadatas"], strict=True)]
            self._bm25 = (data["ids"], BM25(corpus))
            self._bm25_key = key
        return self._bm25

    def search(self, query: str, k: int, mode: str = "hybrid") -> list[Hit]:
        """Top-k chunks. "vector" ranks by embedding similarity only; "hybrid" fuses that ranking
        with BM25 keyword ranking (Reciprocal Rank Fusion). Hit.score is always cosine similarity."""
        if mode not in SEARCH_MODES:
            raise ValueError(f"search mode must be one of {SEARCH_MODES}")
        n = self.count()
        if n == 0:
            return []
        query_vector = self.embed([query], kind="query")[0]
        # Fixed candidate pool so the ranking doesn't depend on k: top-4 is always a prefix of top-10.
        pool = min(n, k if mode == "vector" else max(CANDIDATE_POOL, k))
        vector_ids = self._collection.query(query_embeddings=[query_vector], n_results=pool, include=[])["ids"][0]

        if mode == "vector":
            ids, keyword_ids = vector_ids[:k], []
        else:
            all_ids, bm25 = self._keyword_index()
            keyword_ids = [all_ids[i] for i in bm25.top(query, pool)]
            # Exact identifiers (D13, BR-05) are where embeddings are weakest, so trust keywords more.
            keyword_weight = 2.0 if any(is_identifier(t) for t in tokenize(query)) else 1.0
            ids = reciprocal_rank_fusion([vector_ids, keyword_ids], weights=[1.0, keyword_weight])[:k]

        got = self._collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"], got["embeddings"], strict=True),
                         strict=True))
        q = np.asarray(query_vector, dtype=float)
        hits = []
        for chunk_id in ids:
            doc, meta, emb = by_id[chunk_id]
            e = np.asarray(emb, dtype=float)
            cosine = float(e @ q / ((np.linalg.norm(e) * np.linalg.norm(q)) or 1.0))
            in_vec, in_kw = chunk_id in vector_ids, chunk_id in keyword_ids
            via = "both" if in_vec and in_kw else ("keyword" if in_kw else "meaning")
            hits.append(Hit(text=doc, source=meta["source"], page=int(meta["page"]), score=cosine,
                            section=meta.get("section", ""), via=via))
        return hits
