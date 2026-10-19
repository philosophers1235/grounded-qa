"""Optional dense retrieval.

Nothing here talks to an embeddings provider. To use one, write a factory that
returns an object with a ``name`` and an ``embed(texts) -> array`` method and
pass it as ``--embedder package.module:factory`` (or GQA_EMBEDDER).
"""

import importlib
import os
import sys
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one vector per text, shape (len(texts), dim)."""
        ...


def load_embedder(spec: str) -> Embedder:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"embedder must look like 'package.module:factory', got {spec!r}")
    # The gqa entry point does not put the working directory on sys.path; add it so
    # a module next to your documents can be used without packaging it.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    factory = getattr(importlib.import_module(module_name), attr)
    return factory()


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.where(norms == 0, 1.0, norms)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        query = query_vector / (np.linalg.norm(query_vector) or 1.0)
        similarities = self.vectors @ query
        ranked = np.argsort(-similarities, kind="stable")[:k]
        return [(int(i), float(similarities[i])) for i in ranked]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists of ids; k=60 is the constant from Cormack et al. (2009)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: -pair[1])
