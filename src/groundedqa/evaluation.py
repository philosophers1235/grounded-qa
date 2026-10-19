"""Score retrieval, citations and refusals against a YAML question set."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .chunking import Chunk
from .pipeline import QAService


@dataclass(frozen=True)
class ExpectedSource:
    doc: str
    section: str = ""

    def matches(self, chunk: Chunk) -> bool:
        return chunk.doc_id == self.doc and chunk.heading.lower().startswith(self.section.lower())


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool = True
    sources: tuple[ExpectedSource, ...] = ()
    example: bool = False


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = []
    for item in data.get("questions", []):
        question = Question(
            id=item["id"],
            question=item["question"],
            answerable=item.get("answerable", True),
            sources=tuple(ExpectedSource(s["doc"], s.get("section", "")) for s in item.get("sources", [])),
            example=item.get("example", False),
        )
        if question.answerable != bool(question.sources):
            raise ValueError(f"{path}: question {question.id!r} must list sources if and only if it is answerable")
        questions.append(question)
    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    refused: bool
    best_support: float
    expected_rank: int | None
    cites_expected: bool
    cited: list[str]
    answer: str
    note: str | None = None

    @property
    def refusal_correct(self) -> bool:
        return self.refused != self.answerable


@dataclass
class Report:
    mode: str
    model: str | None
    k: int
    min_support: float
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.answerable]

    def metrics(self) -> dict:
        answerable = self.answerable
        return {
            "questions": len(self.results),
            "answerable": len(answerable),
            "retrieval_hit_at_1": _ratio(sum(r.expected_rank == 1 for r in answerable), len(answerable)),
            "retrieval_hit_at_k": _ratio(sum(r.expected_rank is not None for r in answerable), len(answerable)),
            "citation_correct": _ratio(sum(r.cites_expected for r in answerable), len(answerable)),
            "refusal_accuracy": _ratio(sum(r.refusal_correct for r in self.results), len(self.results)),
            "false_refusals": sum(r.answerable and r.refused for r in self.results),
            "missed_refusals": sum(not r.answerable and not r.refused for r in self.results),
        }

    def to_dict(self, questions_file: Path) -> dict:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "questions_file": str(questions_file),
            "mode": self.mode,
            "model": self.model,
            "k": self.k,
            "min_support": self.min_support,
            "metrics": self.metrics(),
            "results": [asdict(r) for r in self.results],
        }

    def write(self, path: Path, questions_file: Path) -> None:
        path.write_text(json.dumps(self.to_dict(questions_file), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_eval(service: QAService, questions: list[Question], k: int) -> Report:
    report = Report(mode=service.mode, model=None, k=k, min_support=service.retriever.min_support)
    for question in questions:
        answer = service.ask(question.question, top_k=k)
        hits = answer.retrieval.hits if answer.retrieval else []
        expected_rank = next(
            (rank for rank, hit in enumerate(hits, 1) if any(s.matches(hit.chunk) for s in question.sources)),
            None,
        )
        cites_expected = not answer.refused and any(
            expected.matches(source.chunk) for source in answer.sources for expected in question.sources
        )
        report.results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                answerable=question.answerable,
                refused=answer.refused,
                best_support=round(answer.retrieval.best_support, 3) if answer.retrieval else 0.0,
                expected_rank=expected_rank,
                cites_expected=cites_expected,
                cited=[source.chunk.label for source in answer.sources],
                answer=answer.text,
                note=answer.note,
            )
        )
    # Read the mode at the end: rejected credentials switch the service offline mid-run.
    report.mode = service.mode
    report.model = service.settings.model if service.claude is not None else None
    return report


def _ratio(hits: int, total: int) -> dict:
    return {"hits": hits, "total": total, "rate": round(hits / total, 3) if total else None}
