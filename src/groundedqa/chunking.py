"""Split documents into retrievable chunks that never cross a section or page boundary."""

import re
from dataclasses import asdict, dataclass

from .documents import Document, Section
from .text import sentence_spans

DEFAULT_MAX_CHARS = 900
DEFAULT_OVERLAP_CHARS = 200

_PARAGRAPH = re.compile(r"(?:[^\n]|\n(?![ \t]*\n))+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_path: str
    heading: str
    page: int | None
    start: int
    end: int
    text: str

    @property
    def label(self) -> str:
        label = self.doc_title
        if self.heading:
            label += f" — {self.heading}"
        if self.page is not None:
            label += f" (p. {self.page})"
        return label

    @property
    def search_text(self) -> str:
        return f"{self.doc_title}\n{self.heading}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def chunk_documents(
    documents: list[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    return [chunk for doc in documents for chunk in chunk_document(doc, max_chars, overlap_chars)]


def chunk_document(
    doc: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    chunks = []
    for section in doc.sections:
        for start, end in _pack(doc.text, _units(doc.text, section, max_chars), max_chars, overlap_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    doc_path=doc.path,
                    heading=section.heading,
                    page=section.page,
                    start=start,
                    end=end,
                    text=doc.text[start:end],
                )
            )
    return chunks


def _units(text: str, section: Section, max_chars: int) -> list[tuple[int, int]]:
    """Paragraph spans in the section; paragraphs longer than max_chars become sentence spans."""
    units = []
    body = text[section.start : section.end]
    for match in _PARAGRAPH.finditer(body):
        paragraph = match.group()
        if not paragraph.strip():
            continue
        start = section.start + match.start() + len(paragraph) - len(paragraph.lstrip())
        end = section.start + match.end() - (len(paragraph) - len(paragraph.rstrip()))
        if end - start <= max_chars:
            units.append((start, end))
        else:
            units.extend((start + s, start + e) for s, e in sentence_spans(text[start:end]))
    return units


def _pack(text: str, units: list[tuple[int, int]], max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for unit_start, unit_end in units:
        if current is None:
            current = (unit_start, unit_end)
        elif unit_end - current[0] <= max_chars:
            current = (current[0], unit_end)
        else:
            spans.append(current)
            overlap = _overlap_start(text, current, overlap_chars)
            if overlap is not None and unit_end - overlap <= max_chars:
                current = (overlap, unit_end)
            else:
                current = (unit_start, unit_end)
    if current is not None:
        spans.append(current)
    return spans


def _overlap_start(text: str, span: tuple[int, int], overlap_chars: int) -> int | None:
    """Start of the trailing whole sentences of span that fit in overlap_chars, if any.

    The first sentence is never included, otherwise a short chunk would be repeated whole.
    """
    start, end = span
    best = None
    for s, _ in reversed(sentence_spans(text[start:end])[1:]):
        if end - (start + s) > overlap_chars:
            break
        best = start + s
    return best
