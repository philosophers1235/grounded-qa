"""Find the chunks most likely to answer a question, and decide whether any of them really does."""

from dataclasses import dataclass

from .chunking import Chunk
from .dense import Embedder, reciprocal_rank_fusion
from .index import Index
from .text import query_terms

# Dense and lexical candidates are fused from a pool wider than k, so a chunk
# that ranks well in only one list still gets a chance.
CANDIDATE_POOL = 20


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    support: float


@dataclass(frozen=True)
class Retrieval:
    question: str
    terms: list[str]
    hits: list[Hit]
    min_support: float

    @property
    def best_support(self) -> float:
        return max((hit.support for hit in self.hits), default=0.0)

    @property
    def supported(self) -> bool:
        return bool(self.hits) and self.best_support >= self.min_support


class Retriever:
    def __init__(self, index: Index, min_support: float = 0.5, embedder: Embedder | None = None):
        if embedder is not None and index.dense is None:
            raise ValueError("an embedder was given but the index has no vectors; rebuild it with --embedder")
        self.index = index
        self.min_support = min_support
        self.embedder = embedder

    def idf(self, term: str) -> float:
        return self.index.bm25.idf(term)

    def support(self, terms: list[str], chunk_index: int) -> float:
        """Share of the question's information, weighted by IDF, that appears in the chunk.

        A term that never occurs in the corpus gets the highest possible IDF, so a
        question about something the documents never mention scores low everywhere.
        """
        weights = {term: self.idf(term) for term in terms}
        total = sum(weights.values())
        if total == 0:
            return 0.0
        present = self.index.token_sets[chunk_index]
        return sum(weight for term, weight in weights.items() if term in present) / total

    def search(self, question: str, k: int = 5) -> Retrieval:
        terms = query_terms(question)
        lexical = self.index.bm25.top(terms, CANDIDATE_POOL)

        if self.embedder is not None and self.index.dense is not None:
            query_vector = self.embedder.embed([question])[0]
            dense = self.index.dense.search(query_vector, CANDIDATE_POOL)
            ranked = reciprocal_rank_fusion([[i for i, _ in lexical], [i for i, _ in dense]])[:k]
        else:
            ranked = lexical[:k]

        hits = [Hit(self.index.chunks[i], score, self.support(terms, i)) for i, score in ranked]
        return Retrieval(question, terms, hits, self.min_support)
