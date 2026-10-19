"""The answer shape shared by both answerers, the CLI, the evaluation and the web UI."""

from dataclasses import dataclass, field

from .chunking import Chunk
from .retrieval import Retrieval

MODE_CLAUDE = "claude"
MODE_EXTRACTIVE = "extractive"
MODE_LABELS = {MODE_CLAUDE: "Claude", MODE_EXTRACTIVE: "offline extractive mode"}

# Both answerers open a "not in the documents" reply with this sentence, and the
# Claude system prompt asks for it too, so refusals can be detected reliably.
NOT_COVERED = "The documents do not cover this."


@dataclass
class Source:
    number: int
    chunk: Chunk
    highlights: list[tuple[int, int]] = field(default_factory=list)

    def add_highlight(self, start: int, end: int) -> None:
        if (start, end) not in self.highlights and 0 <= start < end <= len(self.chunk.text):
            self.highlights.append((start, end))
            self.highlights.sort()

    @property
    def quotes(self) -> list[str]:
        return [self.chunk.text[start:end] for start, end in self.highlights]

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "doc_path": self.chunk.doc_path,
            "heading": self.chunk.heading,
            "page": self.chunk.page,
            "label": self.chunk.label,
            "text": self.chunk.text,
            "highlights": [list(span) for span in self.highlights],
        }


@dataclass
class AnswerPart:
    text: str
    sources: list[int] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    parts: list[AnswerPart]
    sources: list[Source]
    mode: str
    refused: bool = False
    model: str | None = None
    note: str | None = None
    retrieval: Retrieval | None = None

    @property
    def text(self) -> str:
        """The answer with [n] markers after each cited passage."""
        out = []
        for part in self.parts:
            if not part.sources:
                out.append(part.text)
                continue
            body = part.text.rstrip()
            markers = "".join(f"[{n}]" for n in part.sources)
            out.append(f"{body} {markers}{part.text[len(body):]}")
        return "".join(out).strip()

    @property
    def mode_label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "text": self.text,
            "parts": [{"text": p.text, "sources": p.sources} for p in self.parts],
            "sources": [s.to_dict() for s in self.sources],
            "mode": self.mode,
            "mode_label": self.mode_label,
            "refused": self.refused,
            "model": self.model,
            "note": self.note,
            "retrieved": [
                {"chunk_id": h.chunk.chunk_id, "label": h.chunk.label, "score": round(h.score, 4), "support": round(h.support, 3)}
                for h in (self.retrieval.hits if self.retrieval else [])
            ],
        }


def not_covered(question: str, mode: str, contact: str, retrieval: Retrieval | None = None) -> Answer:
    return Answer(
        question=question,
        parts=[AnswerPart(f"{NOT_COVERED} Please ask {contact}.")],
        sources=[],
        mode=mode,
        refused=True,
        retrieval=retrieval,
    )
