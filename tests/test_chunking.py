from pathlib import Path

import pymupdf

from groundedqa.chunking import chunk_document, chunk_documents
from groundedqa.documents import load_directory, load_document

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "samples/northwind/docs"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_title_and_nested_headings(tmp_path):
    doc = load_document(
        write(
            tmp_path / "guide.md",
            "# Boiler Guide\n\nIntro text.\n\n## Warranty\n\nTwo years.\n\n### Exclusions\n\nFrost.\n\n## Returns\n\n30 days.\n",
        )
    )
    assert doc.title == "Boiler Guide"
    assert [(s.heading, doc.text[s.start : s.end]) for s in doc.sections] == [
        ("", "Intro text."),
        ("Warranty", "Two years."),
        ("Warranty > Exclusions", "Frost."),
        ("Returns", "30 days."),
    ]


def test_long_section_is_split_with_sentence_overlap(tmp_path):
    sentences = [f"Sentence number {i} talks about boiler part {i} in some detail." for i in range(12)]
    doc = load_document(write(tmp_path / "long.md", "# Long\n\n## Body\n\n" + " ".join(sentences) + "\n"))
    chunks = chunk_document(doc, max_chars=200, overlap_chars=80)

    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start < previous.end, "consecutive chunks should share a sentence"
        assert current.text.startswith("Sentence number")


def test_chunks_stay_inside_their_section(tmp_path):
    doc = load_document(write(tmp_path / "two.md", "# T\n\n## A\n\nAlpha text.\n\n## B\n\nBeta text.\n"))
    chunks = chunk_document(doc)
    assert [(c.heading, c.text) for c in chunks] == [("A", "Alpha text."), ("B", "Beta text.")]


def test_chunk_text_matches_document_offsets():
    documents = load_directory(SAMPLE_DOCS)
    by_id = {d.doc_id: d for d in documents}
    for chunk in chunk_documents(documents):
        assert by_id[chunk.doc_id].text[chunk.start : chunk.end] == chunk.text
        assert chunk.text == chunk.text.strip()


def test_pdf_keeps_page_numbers_and_headings(tmp_path):
    path = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        for heading, body in [("Warranty", "The warranty lasts 24 months."), ("Returns", "Parts can be returned within 30 days.")]:
            page = pdf.new_page()
            page.insert_text((72, 80), heading, fontsize=18)
            page.insert_text((72, 120), body, fontsize=11)
            page.insert_text((72, 140), "Keep the receipt.", fontsize=11)
        pdf.set_metadata({"title": "Service Manual"})
        pdf.save(path)

    chunks = chunk_document(load_document(path))
    assert {c.doc_title for c in chunks} == {"Service Manual"}
    assert [(c.page, c.heading) for c in chunks] == [(1, "Warranty"), (2, "Returns")]
    assert "24 months" in chunks[0].text
    assert chunks[1].label == "Service Manual — Returns (p. 2)"
