import anthropic
import pytest
from fastapi.testclient import TestClient
from helpers import FakeClient, api_error

from groundedqa.config import Settings
from groundedqa.pipeline import QAService
from groundedqa.retrieval import Retriever
from groundedqa.web.app import create_app


@pytest.fixture
def offline_client(northwind_index):
    service = QAService(Retriever(northwind_index), Settings(), offline=True)
    return TestClient(create_app(service, examples=["What does error code E10 mean?"]))


def test_home_page_lists_documents_and_examples(offline_client):
    page = offline_client.get("/")
    assert page.status_code == 200
    assert "Ask the documents" in page.text
    assert "Linea 24 Warranty Manual" in page.text
    assert 'data-question="What does error code E10 mean?"' in page.text
    assert "Offline extractive mode" in page.text


def test_static_assets_are_served(offline_client):
    assert offline_client.get("/static/app.js").status_code == 200
    assert offline_client.get("/static/style.css").status_code == 200


def test_ask_returns_parts_and_highlighted_sources(offline_client):
    response = offline_client.post("/api/ask", json={"question": "How much does the Care Plus plan cost?"})
    data = response.json()
    assert response.status_code == 200
    assert data["mode"] == "extractive"
    assert not data["refused"]
    first = data["sources"][0]
    assert first["heading"] == "Maintenance plans"
    assert any("EUR 189" in first["text"][start:end] for start, end in first["highlights"])
    assert all(part["sources"] for part in data["parts"])


def test_ask_rejects_blank_questions(offline_client):
    assert offline_client.post("/api/ask", json={"question": "   "}).status_code == 422
    assert offline_client.post("/api/ask", json={}).status_code == 422


def test_api_errors_are_returned_as_502(northwind_index):
    client = FakeClient(error=api_error(anthropic.InternalServerError, 500))
    service = QAService(Retriever(northwind_index), Settings(), client=client)
    web = TestClient(create_app(service))
    response = web.post("/api/ask", json={"question": "How much does the Care Plus plan cost?"})
    assert response.status_code == 502
    assert "500" in response.json()["detail"]
