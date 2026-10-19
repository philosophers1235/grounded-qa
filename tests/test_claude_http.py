"""Run the real SDK against a local stand-in for the Messages API.

This checks what actually goes over the wire (body and beta header) and that a
response with citations parses through the SDK's own types, without a key.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest
from helpers import make_chunk

from groundedqa.claude import FALLBACK_BETA, ClaudeAnswerer
from groundedqa.retrieval import Hit

WARRANTY = "The standard warranty is 24 months from the date of installation."

RESPONSE = {
    "id": "msg_local",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "The standard warranty is ", "citations": None},
        {
            "type": "text",
            "text": "24 months from installation",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "The standard warranty is 24 months from the date of installation.",
                    "document_index": 0,
                    "document_title": "Manual — Section",
                    "start_char_index": 0,
                    "end_char_index": len(WARRANTY),
                }
            ],
        },
        {"type": "text", "text": ".", "citations": None},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 420, "output_tokens": 31},
}


@pytest.fixture
def local_api():
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "headers": dict(self.headers), "body": body})
            payload = json.dumps(RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_request_and_response_round_trip_through_the_sdk(local_api):
    base_url, received = local_api
    client = anthropic.Anthropic(api_key="local-test-key", base_url=base_url, max_retries=0)
    hits = [Hit(make_chunk(0, WARRANTY), 3.0, 1.0)]

    answer = ClaudeAnswerer(client).answer("How long is the warranty?", hits)

    [request] = received
    assert request["path"].startswith("/v1/messages")
    assert FALLBACK_BETA in request["headers"]["anthropic-beta"]
    body = request["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "low"}
    assert "thinking" not in body and "betas" not in body
    assert body["messages"][0]["content"][0]["citations"] == {"enabled": True}

    assert answer.text == "The standard warranty is 24 months from installation [1]."
    assert answer.sources[0].quotes == [WARRANTY]
    assert not answer.refused
