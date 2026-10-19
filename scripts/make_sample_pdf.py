"""Render the Linea 24 service guide to PDF so the sample corpus has a real PDF in it.

Usage: uv run python scripts/make_sample_pdf.py [SOURCE.md] [OUTPUT.pdf]

Only the Markdown the source file uses is supported: #/## headings, paragraphs,
"- " bullets and "1. " numbered lists.
"""

import html
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "samples/northwind/pdf-source/linea-24-service-guide.md"
DEFAULT_OUTPUT = ROOT / "samples/northwind/docs/linea-24-service-guide.pdf"

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; color: #222; }
h1 { font-size: 20pt; margin: 0 0 14pt 0; }
h2 { font-size: 13pt; margin: 16pt 0 6pt 0; }
p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 7pt 0; }
li { margin: 0 0 3pt 0; }
"""


def markdown_to_html(source: str) -> str:
    out: list[str] = []
    list_tag: str | None = None

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for line in source.splitlines():
        line = line.strip()
        bullet = re.match(r"^- (.*)", line)
        numbered = re.match(r"^\d+\. (.*)", line)
        if bullet or numbered:
            tag = "ul" if bullet else "ol"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{html.escape((bullet or numbered).group(1))}</li>")
            continue
        close_list()
        if not line:
            continue
        if line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        else:
            out.append(f"<p>{html.escape(line)}</p>")
    close_list()
    return "\n".join(out)


def render(source: Path, output: Path) -> int:
    text = source.read_text(encoding="utf-8")
    title = next(line[2:].strip() for line in text.splitlines() if line.startswith("# "))

    story = pymupdf.Story(html=markdown_to_html(text), user_css=CSS)
    page_rect = pymupdf.paper_rect("a4")
    content_rect = page_rect + (60, 60, -60, -60)

    tmp = output.with_suffix(".tmp.pdf")
    writer = pymupdf.DocumentWriter(str(tmp))
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()

    # DocumentWriter cannot set metadata, so reopen and save a final copy.
    with pymupdf.open(tmp) as pdf:
        pdf.set_metadata({"title": title, "author": "Northwind Heating"})
        pdf.save(output, garbage=3, deflate=True)
        pages = pdf.page_count
    tmp.unlink()
    return pages


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    pages = render(source, output)
    print(f"wrote {output} ({pages} pages)")


if __name__ == "__main__":
    main()
