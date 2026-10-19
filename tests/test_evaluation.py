import json

import pytest
from helpers import make_chunk

from groundedqa.config import Settings
from groundedqa.evaluation import ExpectedSource, load_questions, run_eval
from groundedqa.pipeline import QAService
from groundedqa.retrieval import Retriever

QUESTIONS = """
questions:
  - id: length
    question: How long is the standard warranty on the Linea 24 boiler?
    sources:
      - doc: linea-24-warranty
        section: standard warranty
  - id: e10
    question: What does error code E10 mean?
    sources:
      - doc: linea-24-service-guide
        section: Error codes
  - id: wrong-source
    question: How much does the Care Plus plan cost?
    sources:
      - doc: price-list
  - id: solar
    question: Does Northwind offer a warranty on solar thermal panels?
    answerable: false
"""


def test_expected_source_matches_heading_prefix_case_insensitively():
    chunk = make_chunk(0, "text", heading="Standard warranty period")
    assert ExpectedSource("doc", "standard warranty").matches(chunk)
    assert ExpectedSource("doc").matches(chunk)
    assert not ExpectedSource("doc", "warranty period").matches(chunk)
    assert not ExpectedSource("other").matches(chunk)


def test_answerable_questions_must_name_sources(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("questions:\n  - id: a\n    question: Anything?\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sources"):
        load_questions(path)


def test_offline_eval_scores_and_writes_results(tmp_path, northwind_index):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(QUESTIONS, encoding="utf-8")
    service = QAService(Retriever(northwind_index), Settings(), offline=True)

    report = run_eval(service, load_questions(questions_path), k=5)
    metrics = report.metrics()

    assert report.mode == "extractive"
    assert metrics["retrieval_hit_at_k"] == {"hits": 2, "total": 3, "rate": 0.667}
    assert metrics["citation_correct"] == {"hits": 2, "total": 3, "rate": 0.667}
    assert metrics["refusal_accuracy"] == {"hits": 4, "total": 4, "rate": 1.0}
    assert metrics["false_refusals"] == metrics["missed_refusals"] == 0

    out = tmp_path / "results.json"
    report.write(out, questions_path)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["metrics"] == metrics
    assert [r["id"] for r in saved["results"]] == ["length", "e10", "wrong-source", "solar"]
    assert saved["results"][3]["refused"] is True
