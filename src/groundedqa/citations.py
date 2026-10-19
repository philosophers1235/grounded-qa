"""Map the citations in a Messages API response back to the chunks that were sent."""

from .answers import MODE_CLAUDE, NOT_COVERED, Answer, AnswerPart, Source
from .chunking import Chunk


def answer_from_response(question: str, response, chunks: list[Chunk]) -> Answer:
    """Build an Answer from a response whose stop_reason has already been checked.

    ``chunks`` must be in the order their document blocks were sent: a citation's
    ``document_index`` points into that list, and for plain-text documents its
    character indices point into the chunk text.
    """
    parts: list[AnswerPart] = []
    sources: dict[int, Source] = {}

    for block in response.content:
        # Thinking blocks (empty by default on this model) and fallback markers carry no answer text.
        if block.type != "text":
            continue
        numbers: list[int] = []
        for citation in block.citations or []:
            index = citation.document_index
            if not 0 <= index < len(chunks):
                continue
            source = sources.get(index)
            if source is None:
                source = sources[index] = Source(number=len(sources) + 1, chunk=chunks[index])
            _highlight(source, citation)
            if source.number not in numbers:
                numbers.append(source.number)
        parts.append(AnswerPart(block.text, numbers))

    text = "".join(part.text for part in parts).strip()
    refused = not sources or text.startswith(NOT_COVERED)
    return Answer(
        question=question,
        parts=parts,
        sources=sorted(sources.values(), key=lambda s: s.number),
        mode=MODE_CLAUDE,
        refused=refused,
        model=getattr(response, "model", None),
    )


def _highlight(source: Source, citation) -> None:
    cited = (getattr(citation, "cited_text", "") or "").strip()
    if citation.type == "char_location":
        start, end = citation.start_char_index, citation.end_char_index
        if source.chunk.text[start:end].strip() == cited or not cited:
            source.add_highlight(start, end)
            return
    # Fall back to locating the quoted text, in case the indices and our text disagree.
    if cited:
        position = source.chunk.text.find(cited)
        if position >= 0:
            source.add_highlight(position, position + len(cited))
