"""Keyword search (Okapi BM25) and Reciprocal Rank Fusion, with no extra dependencies.

Embeddings capture meaning but blur exact identifiers: "decision D13" and "decision D21" look
almost identical to an embedding model. BM25 scores exact token overlap, so combining both
rankings (hybrid search) finds IDs, codes and rare names while keeping semantic matching.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# Words joined by - . : stay together as one identifier token ("br-05", "v2.1") and are also split.
_TOKEN = re.compile(r"\w+(?:[-.:]\w+)*", re.UNICODE)
_SPLIT = re.compile(r"[-.:]")
STOPWORDS = frozenset(
    "a an and are as at be by can do does did for from has have how i in is it its me my of on or "
    "should that the their this to was what when where which who why will with about".split()
)


IDENTIFIER_BOOST = 3.0  # query terms like "d13", "br-05", "v2.1" are what the user is being exact about


def is_identifier(token: str) -> bool:
    """Exact codes: digits mixed with letters (d13, br-05, v2.1) or digits joined by - . :
    (version 0.10.0, date 2026-10-08, python 3.9). Plain numbers ("500") and words are not."""
    has_digit = any(c.isdigit() for c in token)
    return has_digit and (any(c.isalpha() for c in token) or bool(_SPLIT.search(token)))


def identifiers(text: str) -> set[str]:
    """Whole exact codes in a text ("v2.14.0", "0.10.0", "d13"), never their split parts ("v2")."""
    return {m.group() for m in _TOKEN.finditer(text.lower()) if is_identifier(m.group())}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group()
        parts = [p for p in _SPLIT.split(token) if p]
        if len(parts) > 1:
            tokens.append(token)
        tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        weight = {t: IDENTIFIER_BOOST if is_identifier(t) else 1.0 for t in terms}
        result = []
        for tf, length in zip(self.term_freqs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
            result.append(sum(weight[t] * self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                              for t in terms if t in tf))
        return result

    def top(self, query: str, n: int) -> list[int]:
        """Indices of the `n` best documents that share at least one term with the query."""
        scores = self.scores(query)
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
        return ranked[:n]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[str]:
    """Merge ranked id lists: score(id) = sum(weight / (k + rank)). Robust because it only uses
    ranks, so BM25 scores and cosine similarities never need to be put on the same scale."""
    weights = weights or [1.0] * len(rankings)
    scores: defaultdict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] += weight / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
