from helpers import make_chunk, make_message, text_block

from groundedqa.citations import answer_from_response

WARRANTY = "The standard warranty is 24 months from the date of installation. Register within 30 days."
EXTENSION = "The warranty can be extended to 5 years if the boiler is serviced every year."


def test_citations_become_numbered_sources_in_order_of_first_use():
    chunks = [make_chunk(0, WARRANTY), make_chunk(1, EXTENSION)]
    response = make_message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            text_block("The warranty lasts 24 months", (0, WARRANTY, "The standard warranty is 24 months from the date of installation.")),
            text_block(" and can be extended to 5 years", (1, EXTENSION, "The warranty can be extended to 5 years")),
            text_block(" with yearly servicing.", (1, EXTENSION, "if the boiler is serviced every year")),
        ]
    )

    answer = answer_from_response("How long is the warranty?", response, chunks)

    assert not answer.refused
    assert [s.chunk.chunk_id for s in answer.sources] == ["doc#0", "doc#1"]
    assert [p.sources for p in answer.parts] == [[1], [2], [2]]
    assert answer.text == "The warranty lasts 24 months [1] and can be extended to 5 years [2] with yearly servicing. [2]"
    assert answer.sources[0].quotes == ["The standard warranty is 24 months from the date of installation."]
    assert answer.sources[1].quotes == ["The warranty can be extended to 5 years", "if the boiler is serviced every year"]


def test_cited_document_index_is_relative_to_what_was_sent():
    chunks = [make_chunk(0, EXTENSION), make_chunk(7, WARRANTY)]
    response = make_message([text_block("24 months.", (1, WARRANTY, "24 months"))])
    answer = answer_from_response("q", response, chunks)
    assert answer.sources[0].chunk.chunk_id == "doc#7"
    assert answer.sources[0].number == 1


def test_answer_without_citations_counts_as_refused():
    response = make_message([text_block("I think it is probably two years.")])
    answer = answer_from_response("q", response, [make_chunk(0, WARRANTY)])
    assert answer.refused
    assert answer.sources == []


def test_not_covered_reply_is_refused_even_if_it_cites_a_contact():
    contact = "Write to support@example.com for anything not covered here."
    response = make_message(
        [
            text_block("The documents do not cover this. "),
            text_block("You can write to support@example.com.", (0, contact, "Write to support@example.com")),
        ]
    )
    answer = answer_from_response("q", response, [make_chunk(0, contact)])
    assert answer.refused


def test_highlight_falls_back_to_cited_text_when_indices_disagree():
    chunks = [make_chunk(0, WARRANTY)]
    block = text_block("Register within 30 days.", (0, WARRANTY, "Register within 30 days."))
    block["citations"][0]["start_char_index"] = 0
    block["citations"][0]["end_char_index"] = 10
    answer = answer_from_response("q", make_message([block]), chunks)
    assert answer.sources[0].quotes == ["Register within 30 days."]


def test_out_of_range_document_index_is_ignored():
    block = text_block("Something.", (0, WARRANTY, "24 months"))
    block["citations"][0]["document_index"] = 5
    answer = answer_from_response("q", make_message([block]), [make_chunk(0, WARRANTY)])
    assert answer.sources == []
    assert answer.refused
