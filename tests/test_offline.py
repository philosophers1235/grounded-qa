import anthropic
from helpers import FakeClient, api_error, make_message, text_block

from groundedqa.answers import MODE_CLAUDE, MODE_EXTRACTIVE, NOT_COVERED
from groundedqa.config import Settings
from groundedqa.extractive import answer_extractively
from groundedqa.pipeline import NO_CREDENTIALS, QAService
from groundedqa.retrieval import Retriever


def test_extractive_answer_quotes_the_right_passage(northwind_index):
    retriever = Retriever(northwind_index)
    retrieval = retriever.search("How long is the standard warranty on the Linea 24 boiler?")
    answer = answer_extractively(retrieval, retriever.idf, contact="support")

    assert answer.mode == MODE_EXTRACTIVE
    assert not answer.refused
    first = answer.sources[0]
    assert (first.chunk.doc_id, first.chunk.heading) == ("linea-24-warranty", "Standard warranty period")
    assert "24 months from the date of installation" in first.quotes[0]


def test_extractive_answer_only_contains_quoted_text(northwind_index):
    retriever = Retriever(northwind_index)
    answer = answer_extractively(retriever.search("What does error code E10 mean?"), retriever.idf, contact="support")
    assert answer.sources[0].chunk.page == 3
    for part in answer.parts:
        [number] = part.sources
        quotes = [" ".join(q.split()).lower() for q in answer.sources[number - 1].quotes]
        assert part.text.strip().rstrip(".").lower() in [q.rstrip(".;") for q in quotes]


def test_offline_service_declines_questions_the_documents_do_not_cover(northwind_index):
    service = QAService(Retriever(northwind_index), Settings(contact="Northwind support"), offline=True)
    answer = service.ask("Can I control the Linea 24 from a smartphone app?")
    assert answer.refused
    assert answer.sources == []
    assert answer.text == f"{NOT_COVERED} Please ask Northwind support."
    assert answer.note is None


def test_without_credentials_the_service_answers_offline_and_says_why(northwind_index):
    service = QAService(Retriever(northwind_index), Settings())
    answer = service.ask("How much does the Care Plus plan cost?")
    assert service.mode == MODE_EXTRACTIVE
    assert answer.note == NO_CREDENTIALS
    assert "EUR 189" in answer.text


def test_rejected_credentials_switch_the_session_to_offline(northwind_index):
    client = FakeClient(error=api_error(anthropic.AuthenticationError, 401))
    service = QAService(Retriever(northwind_index), Settings(), client=client)
    assert service.mode == MODE_CLAUDE

    first = service.ask("How much does the Care Plus plan cost?")
    second = service.ask("What does error code E10 mean?")

    assert len(client.calls) == 1
    assert first.mode == second.mode == MODE_EXTRACTIVE
    assert "rejected the credentials" in first.note
    assert not first.refused


def test_unsupported_questions_never_reach_claude(northwind_index):
    client = FakeClient(make_message([text_block("unused")]))
    service = QAService(Retriever(northwind_index), Settings(), client=client)
    answer = service.ask("Who is the managing director of Northwind Heating?")
    assert answer.refused
    assert client.calls == []
