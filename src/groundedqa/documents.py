"""Load .md, .txt and .pdf files into one structure: full text plus headed sections."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")

_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_PDF_BULLET = re.compile(r"^[•●▪–-]\s+")
_PDF_NUMBERED = re.compile(r"^\d+\.\s")
# Default extraction flags minus TEXT_PRESERVE_LIGATURES, so "ﬂue" comes out as "flue".
_PDF_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
HEADING_PATH_SEPARATOR = " > "


@dataclass(frozen=True)
class Section:
    heading: str
    start: int
    end: int
    page: int | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    path: str
    text: str
    sections: list[Section]


def load_directory(root: Path) -> list[Document]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p, doc_id=p.relative_to(root).with_suffix("").as_posix()) for p in files]


def load_document(path: Path, doc_id: str | None = None) -> Document:
    doc_id = doc_id or path.stem
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, doc_id)
    if suffix in (".md", ".markdown"):
        return _load_markdown(path, doc_id)
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        sections = _trimmed_sections(text, [("", 0, len(text), None)])
        return Document(doc_id, _title_from_stem(path), str(path), text, sections)
    raise ValueError(f"unsupported file type: {path}")


def _load_markdown(path: Path, doc_id: str) -> Document:
    text = path.read_text(encoding="utf-8")
    headings = list(_MD_HEADING.finditer(text))
    title = next((m.group(2) for m in headings if len(m.group(1)) == 1), None) or _title_from_stem(path)

    raw_sections = []
    trail: list[tuple[int, str]] = []
    heading, body_start = "", 0
    for match in headings:
        raw_sections.append((heading, body_start, match.start(), None))
        level = len(match.group(1))
        trail = [(lvl, h) for lvl, h in trail if lvl < level] + [(level, match.group(2))]
        # The H1 is the document title; repeating it in every section label adds nothing.
        heading = HEADING_PATH_SEPARATOR.join(h for lvl, h in trail if lvl > 1)
        body_start = match.end()
    raw_sections.append((heading, body_start, len(text), None))

    return Document(doc_id, title, str(path), text, _trimmed_sections(text, raw_sections))


def _load_pdf(path: Path, doc_id: str) -> Document:
    paragraphs: list[tuple[str, float, int]] = []
    with pymupdf.open(path) as pdf:
        meta_title = (pdf.metadata or {}).get("title", "").strip()
        for page in pdf:
            for block in page.get_text("dict", flags=_PDF_FLAGS)["blocks"]:
                if block.get("type") == 0:
                    text, size = _pdf_block_text(block)
                    if text:
                        paragraphs.append((text, size, page.number + 1))

    if not paragraphs:
        raise ValueError(f"no extractable text in {path} (scanned PDFs need OCR first)")

    size_weights: Counter[float] = Counter()
    for text, size, _ in paragraphs:
        size_weights[size] += len(text)
    body_size = size_weights.most_common(1)[0][0]

    def is_heading(text: str, size: float) -> bool:
        return size >= body_size * 1.15 and len(text) < 120 and "\n" not in text

    title = meta_title or next((t for t, s, _ in paragraphs if is_heading(t, s)), "") or _title_from_stem(path)

    parts: list[str] = []
    raw_sections = []
    heading, body_start, current_page, pos = "", 0, paragraphs[0][2], 0
    for text, size, page in paragraphs:
        heading_here = is_heading(text, size)
        if heading_here or page != current_page:
            raw_sections.append((heading, body_start, max(pos - 2, body_start), current_page))
            body_start, current_page = pos, page
        parts.append(text)
        pos += len(text) + 2
        if heading_here:
            heading = "" if text == title else text
            body_start = pos
    full_text = "\n\n".join(parts)
    raw_sections.append((heading, body_start, len(full_text), current_page))

    return Document(doc_id, title, str(path), full_text, _trimmed_sections(full_text, raw_sections))


def _pdf_block_text(block: dict) -> tuple[str, float]:
    """Join a text block's wrapped lines back into paragraph text, one list item per line."""
    lines: list[str] = []
    sizes: Counter[float] = Counter()
    for line in block["lines"]:
        text = "".join(span["text"] for span in line["spans"]).strip()
        for span in line["spans"]:
            sizes[round(span["size"], 1)] += len(span["text"])
        if not text:
            continue
        bullet = _PDF_BULLET.match(text)
        if bullet:
            lines.append("- " + text[bullet.end():])
        elif _PDF_NUMBERED.match(text) or not lines:
            lines.append(text)
        elif lines[-1].endswith("-") and lines[-1][-2:-1].isalpha():
            lines[-1] += text
        else:
            lines[-1] += " " + text
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    return "\n".join(lines), size


def _trimmed_sections(text: str, raw: list[tuple[str, int, int, int | None]]) -> list[Section]:
    sections = []
    for heading, start, end, page in raw:
        body = text[start:end]
        if not body.strip():
            continue
        lead = len(body) - len(body.lstrip())
        trail = len(body) - len(body.rstrip())
        sections.append(Section(heading, start + lead, end - trail, page))
    return sections


def _title_from_stem(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip().capitalize()
