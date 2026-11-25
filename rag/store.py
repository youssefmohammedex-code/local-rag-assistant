"""Chroma vector store with embeddings computed locally by Ollama."""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, replace

import chromadb
import numpy as np
import ollama

from rag.bm25 import BM25, identifiers, reciprocal_rank_fusion
from rag.chunking import Chunk
from rag.config import Settings

SEARCH_MODES = ("hybrid", "vector")
CANDIDATE_POOL = 50  # candidates taken from each ranking before fusion
_ORDER = re.compile(r"::s(\d+)::c(\d+)$")  # section index and chunk index inside a chunk id


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better
    section: str = ""
    via: str = "meaning"  # "meaning" (vector), "keyword" (BM25), "both", or "section" (list expansion)
    id: str = ""

    def truncated(self, max_chars: int) -> Hit:
        if len(self.text) <= max_chars:
            return self
        return replace(self, text=self.text[:max_chars].rsplit(" ", 1)[0] + " …")


class VectorStore:
    def __init__(self, settings: Settings, collection_name: str | None = None):
        self.settings = settings
        self.name = collection_name or settings.collection_name
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.db_dir))
        self._ollama = ollama.Client(host=settings.ollama_host)
        self._collection = self._get_collection()
        self._bm25: tuple[list[str], BM25] | None = None
        self._heading_ids: dict[str, set[str]] = {}
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
            self._heading_ids = {cid: identifiers(m.get("section", "").split(" > ")[-1])
                                 for cid, m in zip(data["ids"], data["metadatas"], strict=True)}
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
            # Exact identifiers (D13, BR-05, 0.10.0) are where embeddings are weakest, so trust keywords more.
            query_ids = identifiers(query)
            keyword_weight = 2.0 if query_ids else 1.0
            fused = reciprocal_rank_fusion([vector_ids, keyword_ids], weights=[1.0, keyword_weight])
            # A section whose own heading carries the exact code asked about ("0.10.0", "D13 — ...")
            # goes first. Fusion alone favours passages that appear in both rankings, so many
            # near-misses (0.10.1, 0.10.2, ...) outranked the exact match.
            pinned = [cid for cid in keyword_ids if query_ids & self._heading_ids.get(cid, set())]
            ids = (pinned + [cid for cid in fused if cid not in pinned])[:k]

        via = {i: "both" if i in vector_ids and i in keyword_ids else ("keyword" if i in keyword_ids else "meaning")
               for i in ids}
        return self._hits(ids, query_vector, via)

    def _hits(self, ids: list[str], query_vector: list[float], via: dict[str, str]) -> list[Hit]:
        """Build Hits in the given order, scoring each by cosine similarity to the query."""
        got = self._collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"], got["embeddings"], strict=True),
                         strict=True))
        q = np.asarray(query_vector, dtype=float)
        hits = []
        for chunk_id in ids:
            doc, meta, emb = by_id[chunk_id]
            e = np.asarray(emb, dtype=float)
            cosine = float(e @ q / ((np.linalg.norm(e) * np.linalg.norm(q)) or 1.0))
            hits.append(Hit(text=doc, source=meta["source"], page=int(meta["page"]), score=cosine,
                            section=meta.get("section", ""), via=via[chunk_id], id=chunk_id))
        return hits

    def outline(self, source: str) -> list[tuple[str, str]]:
        """(chunk_id, section path) for every chunk of `source`, in document order."""
        data = self._collection.get(where={"source": source}, include=["metadatas"])
        rows = []
        for chunk_id, meta in zip(data["ids"], data["metadatas"], strict=True):
            match = _ORDER.search(chunk_id)
            order = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
            rows.append((order, chunk_id, meta.get("section", "")))
        return [(chunk_id, section) for _, chunk_id, section in sorted(rows)]

    def hits_for(self, ids: list[str], query: str, via: str) -> list[Hit]:
        if not ids:
            return []
        return self._hits(ids, self.embed([query], kind="query")[0], dict.fromkeys(ids, via))

    def similarities(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity between a query and short texts, using the same embedding model."""
        q = np.asarray(self.embed([query], kind="query")[0], dtype=float)
        vectors = np.asarray(self.embed(texts, kind="document"), dtype=float)
        norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(q) or 1.0)
        return [float(s) for s in (vectors @ q) / np.where(norms == 0, 1.0, norms)]


def children_in_outline(outline: list[tuple[str, str]], parent: str) -> list[tuple[str, str]]:
    """Direct sub-sections of `parent` as (name, first chunk id), in document order.
    parent "" means the top level of the file, for documents without a single title heading."""
    first: dict[str, str] = {}
    for chunk_id, section in outline:
        if not section:
            continue
        if parent:
            if not section.startswith(parent + " > "):
                continue
            child = section[len(parent) + 3:].split(" > ")[0]
        else:
            child = section.split(" > ")[0]
        first.setdefault(child, chunk_id)
    return list(first.items())
