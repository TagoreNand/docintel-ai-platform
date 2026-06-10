"""Hand-rolled BM25 Okapi sparse retrieval.

Implementing BM25 directly (rather than pulling a dependency) keeps the lexical
half of the hybrid retriever transparent and persist-free: the index is cheap to
rebuild from the chunk corpus and is cached in-process, keyed by a corpus
signature so it is only recomputed when documents change.
"""

from __future__ import annotations

import math
import re
import threading

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "this", "that", "it", "from",
    "which", "who", "what", "when", "where", "how", "into", "than", "then",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _ENGLISH_STOPWORDS]
    return tokens


class BM25Okapi:
    """Classic BM25 Okapi ranking function."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in corpus_tokens:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1
        # BM25 idf with the standard +0.5 smoothing (floored at a small epsilon).
        self.idf = {
            token: max(1e-6, math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5)))
            for token, n in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not self.corpus_size or self.avgdl == 0.0:
            return scores
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(token, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class SparseIndex:
    """BM25 over a list of chunk records, rebuilt lazily on corpus change."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()
        self._signature: tuple | None = None
        self._records: list = []
        self._bm25: BM25Okapi | None = None

    def build(self, records: list, signature: tuple) -> None:
        with self._lock:
            corpus = [tokenize(r.text) for r in records]
            self._records = records
            self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
            self._signature = signature

    def is_current(self, signature: tuple) -> bool:
        return self._signature == signature

    def invalidate(self) -> None:
        with self._lock:
            self._signature = None

    def search(self, query: str, top_k: int) -> list[tuple]:
        """Return ``[(record, score), ...]`` for the highest scoring chunks."""
        with self._lock:
            if not self._bm25 or not self._records:
                return []
            scores = self._bm25.get_scores(tokenize(query))
            if not np.any(scores > 0):
                return []
            k = min(top_k, len(self._records))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._records[i], float(scores[i])) for i in top if scores[i] > 0]
