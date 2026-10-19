"""Okapi BM25 over pre-tokenised documents."""

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.lengths = np.array([len(tokens) for tokens in documents], dtype=np.float64)
        self.avg_length = float(self.lengths.mean()) if self.n_docs else 0.0

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for doc_index, tokens in enumerate(documents):
            for term, count in Counter(tokens).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(doc_index)
                tfs.append(count)
        self._postings = {
            term: (np.array(ids, dtype=np.int64), np.array(tfs, dtype=np.float64)) for term, (ids, tfs) in postings.items()
        }

    def document_frequency(self, term: str) -> int:
        posting = self._postings.get(term)
        return 0 if posting is None else len(posting[0])

    def idf(self, term: str) -> float:
        # The "+1" variant (as in Lucene) never goes negative for very common terms.
        df = self.document_frequency(term)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, terms: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self.lengths / self.avg_length)
        for term in dict.fromkeys(terms):
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tfs = posting
            scores[ids] += self.idf(term) * tfs * (self.k1 + 1.0) / (tfs + norm[ids])
        return scores

    def top(self, terms: list[str], k: int) -> list[tuple[int, float]]:
        scores = self.scores(terms)
        ranked = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]
