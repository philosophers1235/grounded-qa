import anthropic
import pytest
from helpers import FakeClient, api_error, make_chunk, make_message, text_block

from groundedqa.claude import FALLBACK_BETA, ClaudeAnswerer, ClaudeError, ClaudeUnavailable
from groundedqa.retrieval import Hit

WARRANTY = "The standard warranty is 24 months from the date of installation."


def hits() -> list[Hit]:
    return [
        Hit(make_chunk(0, WARRANTY, heading="Standard warranty period"), 3.2, 1.0),
        Hit(make_chunk(1, "Manuals are printed on paper.", page=4), 0.4, 0.2),
    ]


def test_request_sends_chunks_as_citable_documents_then_the_question():
    request = ClaudeAnswerer(FakeClient(), model="claude-opus-5-5").build_request("How long is the warranty?", hits())

    assert request["model"] == "claude-opus-5-5"
    assert request["max_tokens"] == 2000
    [message] = request["messages"]
    assert message["role"] == "user"
    *documents, question = message["content"]
    assert question == {"type": "text", "text": "How long is the warranty?"}
    assert documents[0] == {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": WARRANTY},
        "title": "Manual — Standard warranty period",
        "citations": {"enabled": True},
    }
    assert documents[1]["title"] == "Manual — Section (p. 4)"


def test_request_sets_effort_only_and_opts_into_fallbacks():
    request = ClaudeAnswerer(FakeClient()).build_request("q", hits())
    assert request["output_config"] == {"effort": "low"}
    assert "thinking" not in request
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert "The documents do not cover this." in request["system"]


def test_answer_calls_the_beta_messages_endpoint():
    response = make_message([text_block("24 months.", (0, WARRANTY, "24 months"))])
    client = FakeClient(response)
    answer = ClaudeAnswerer(client).answer("How long?", hits())
    assert len(client.calls) == 1
    assert answer.sources[0].chunk.heading == "Standard warranty period"
    assert answer.note is None


def test_refusal_stop_reason_discards_content():
    response = make_message(
        [text_block("partial", (0, WARRANTY, "24 months"))],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", hits())
    assert answer.refused
    assert answer.sources == []
    assert "cyber" in answer.note


def test_max_tokens_keeps_the_answer_and_says_it_was_cut():
    response = make_message([text_block("24 months", (0, WARRANTY, "24 months"))], stop_reason="max_tokens")
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", hits())
    assert not answer.refused
    assert "cut off" in answer.note


def test_server_side_fallback_is_reported():
    response = make_message(
        [text_block("24 months", (0, WARRANTY, "24 months"))],
        model="claude-opus-5",
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "iterations": [
                {
                    "type": "fallback_message",
                    "model": "claude-opus-5",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            ],
        },
    )
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", hits())
    assert answer.note == "Answered by claude-opus-5 after a server-side fallback."


def test_unexpected_stop_reason_is_an_error():
    response = make_message([{"type": "text", "text": "", "citations": None}], stop_reason="pause_turn")
    with pytest.raises(ClaudeError, match="pause_turn"):
        ClaudeAnswerer(FakeClient(response)).answer("q", hits())


def test_authentication_error_means_unavailable():
    client = FakeClient(error=api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(ClaudeUnavailable):
        ClaudeAnswerer(client).answer("q", hits())


def test_rate_limit_error_mentions_retry_after():
    client = FakeClient(error=api_error(anthropic.RateLimitError, 429, {"retry-after": "30"}))
    with pytest.raises(ClaudeError, match="Retry after 30"):
        ClaudeAnswerer(client).answer("q", hits())


def test_other_status_errors_are_reported():
    client = FakeClient(error=api_error(anthropic.InternalServerError, 500))
    with pytest.raises(ClaudeError, match="500"):
        ClaudeAnswerer(client).answer("q", hits())
