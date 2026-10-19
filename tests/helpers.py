"""Fake Messages API responses built from the SDK's own types, so field names are checked."""

import anthropic
import httpx2
from anthropic.types.beta import BetaMessage

from groundedqa.chunking import Chunk


def make_chunk(n: int, text: str, heading: str = "Section", page: int | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"doc#{n}",
        doc_id="doc",
        doc_title="Manual",
        doc_path="docs/manual.md",
        heading=heading,
        page=page,
        start=0,
        end=len(text),
        text=text,
    )


def text_block(text: str, *cites: tuple[int, str, str]) -> dict:
    """A text block; each cite is (document_index, document_text, quoted_text)."""
    citations = []
    for index, document_text, quote in cites:
        start = document_text.index(quote)
        citations.append(
            {
                "type": "char_location",
                "cited_text": quote,
                "document_index": index,
                "document_title": "Manual",
                "start_char_index": start,
                "end_char_index": start + len(quote),
            }
        )
    return {"type": "text", "text": text, "citations": citations or None}


def make_message(content: list[dict], stop_reason: str = "end_turn", **extra) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            **extra,
        }
    )


def api_error(cls: type[anthropic.APIStatusError], status: int, headers: dict | None = None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx2.Response(status, headers=headers or {}, request=request), body=None)


class FakeMessages:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeBeta:
    def __init__(self, messages: FakeMessages):
        self.messages = messages


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.messages_api = FakeMessages(response, error)
        self.beta = FakeBeta(self.messages_api)

    @property
    def calls(self) -> list[dict]:
        return self.messages_api.calls
