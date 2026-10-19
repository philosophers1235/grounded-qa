"""A single-page web UI over the same QAService the CLI uses."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..claude import ClaudeError
from ..pipeline import QAService

HERE = Path(__file__).parent


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def create_app(service: QAService, examples: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="grounded-qa", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    documents = service.retriever.index.manifest.get("documents", [])

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "documents": documents,
                "chunk_count": len(service.retriever.index.chunks),
                "mode": service.mode,
                "model": service.settings.model,
                "offline_note": service.offline_note,
                "examples": examples or [],
            },
        )

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Type a question first.")
        try:
            return service.ask(question).to_dict()
        except ClaudeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
