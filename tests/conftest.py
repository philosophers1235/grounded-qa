from pathlib import Path

import pytest

from groundedqa.index import build_index

SAMPLES = Path(__file__).resolve().parent.parent / "samples/northwind"


@pytest.fixture(scope="session")
def northwind_index():
    return build_index(SAMPLES / "docs")


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch):
    # Tests must never reach the paid API, whatever the developer has configured.
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("groundedqa.pipeline.make_client", lambda: None)
