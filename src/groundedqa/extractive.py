"""Offline answers: the best-supported sentences from the top chunks, quoted verbatim.

No model is involved, so nothing is paraphrased and nothing can be invented. The
price is that answers read like excerpts, and a question that needs two facts
combined only gets the excerpts that contain them.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from .answers import MODE_EXTRACTIVE, Answer, AnswerPart, Source, not_covered
from .retrieval import Hit, Retrieval
from .text import sentence_spans, strip_list_marker, tokenize

MAX_CHUNKS = 3
MAX_SENTENCES = 3
# Keep runner-up sentences only if they score at least this share of the best one.
RELATIVE_CUTOFF = 0.6

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class _Candidate:
    score: float
    rank: int
    start: int
    end: int
    hit: Hit


def answer_extractively(retrieval: Retrieval, idf: Callable[[str], float], contact: str) -> Answer:
    weights = {term: idf(term) for term in retrieval.terms}
    total = sum(weights.values())
    candidates = _score_sentences(retrieval.hits[:MAX_CHUNKS], weights, total) if total else []
    if not candidates or candidates[0].score <= 0:
        return not_covered(retrieval.question, MODE_EXTRACTIVE, contact, retrieval)

    chosen = _choose(candidates)
    sources: dict[str, Source] = {}
    parts = []
    for candidate in chosen:
        chunk = candidate.hit.chunk
        source = sources.get(chunk.chunk_id)
        if source is None:
            source = sources[chunk.chunk_id] = Source(number=len(sources) + 1, chunk=chunk)
        sentence = chunk.text[candidate.start : candidate.end]
        without_marker = strip_list_marker(sentence)
        source.add_highlight(candidate.start + len(sentence) - len(without_marker), candidate.end)
        parts.append(AnswerPart(_as_sentence(without_marker) + " ", [source.number]))

    return Answer(
        question=retrieval.question,
        parts=parts,
        sources=list(sources.values()),
        mode=MODE_EXTRACTIVE,
        retrieval=retrieval,
    )


def _score_sentences(hits: list[Hit], weights: dict[str, float], total: float) -> list[_Candidate]:
    """Score each sentence by the share of query weight it covers.

    A sentence is read in the context of its section heading, so "E10, low water
    pressure" under "Error codes" covers a question about "error code E10". It
    must still contain at least one query term itself, or every sentence in the
    section would qualify.
    """
    candidates = []
    for rank, hit in enumerate(hits):
        heading_terms = set(tokenize(hit.chunk.heading))
        for start, end in sentence_spans(hit.chunk.text):
            sentence_terms = set(tokenize(hit.chunk.text[start:end]))
            if not sentence_terms.intersection(weights):
                continue
            covered = sentence_terms | heading_terms
            score = sum(w for t, w in weights.items() if t in covered) / total
            candidates.append(_Candidate(score, rank, start, end, hit))
    candidates.sort(key=lambda c: (-c.score, c.rank, c.start))
    return candidates


def _as_sentence(text: str) -> str:
    """List items read better on their own with a capital letter and a full stop."""
    text = _SPACE.sub(" ", text).strip()
    if text.endswith(";"):
        text = text[:-1] + "."
    return text[:1].upper() + text[1:]


def _choose(candidates: list[_Candidate]) -> list[_Candidate]:
    best = candidates[0].score
    chosen: list[_Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if len(chosen) == MAX_SENTENCES or candidate.score < best * RELATIVE_CUTOFF:
            break
        # Overlapping chunks repeat sentences; quote each one once.
        key = _SPACE.sub(" ", candidate.hit.chunk.text[candidate.start : candidate.end]).lower()
        if key not in seen:
            seen.add(key)
            chosen.append(candidate)
    return chosen
