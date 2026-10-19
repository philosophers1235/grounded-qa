"""Terminal output for answers and evaluation reports."""

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .answers import MODE_CLAUDE, MODE_LABELS, Answer
from .evaluation import Report


def print_answer(console: Console, answer: Answer, show_retrieval: bool = False) -> None:
    if answer.refused:
        title = f"Not in the documents · {answer.mode_label}"
        border = "yellow"
    else:
        title = f"Answer · {answer.mode_label}"
        if answer.mode == MODE_CLAUDE and answer.model:
            title += f" ({answer.model})"
        border = "green" if answer.mode == MODE_CLAUDE else "cyan"

    console.print(Panel(Text(answer.text), title=title, title_align="left", border_style=border, padding=(1, 2)))

    for source in answer.sources:
        heading = Text(f"[{source.number}] ", style="bold")
        heading.append(source.chunk.label)
        details = [Text(source.chunk.doc_path, style="dim")]
        details += [Text(f'"{" ".join(quote.split())}"', style="italic") for quote in source.quotes]
        console.print(Group(heading, Padding(Group(*details), (0, 0, 0, 4))))

    if answer.note:
        console.print(Text(f"Note: {answer.note}", style="yellow"))

    if show_retrieval and answer.retrieval is not None:
        console.print(retrieval_table(answer))


def retrieval_table(answer: Answer) -> Table:
    retrieval = answer.retrieval
    table = Table(
        title=f"Retrieved (best support {retrieval.best_support:.2f}, threshold {retrieval.min_support:.2f})",
        title_justify="left",
        show_edge=False,
    )
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("support", justify="right")
    table.add_column("chunk")
    for rank, hit in enumerate(retrieval.hits, 1):
        table.add_row(str(rank), f"{hit.score:.3f}", f"{hit.support:.2f}", hit.chunk.label)
    return table


def print_report(console: Console, report: Report) -> None:
    metrics = report.metrics()
    unanswerable = metrics["questions"] - metrics["answerable"]
    mode = MODE_LABELS.get(report.mode, report.mode)
    if report.model:
        mode += f" ({report.model})"
    console.print(f"{mode[:1].upper()}{mode[1:]}, k={report.k}, min support {report.min_support:.2f}")
    console.print(
        f"{metrics['questions']} questions: {metrics['answerable']} answerable, "
        f"{unanswerable} not covered by the documents\n"
    )

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
    table.add_column("metric")
    table.add_column("", justify="right")
    table.add_column("", justify="right")
    for label, key in [
        ("retrieval hit@1", "retrieval_hit_at_1"),
        (f"retrieval hit@{report.k}", "retrieval_hit_at_k"),
        ("citation correctness", "citation_correct"),
        ("refusal accuracy", "refusal_accuracy"),
    ]:
        ratio = metrics[key]
        rate = f"{ratio['rate']:.1%}" if ratio["rate"] is not None else "-"
        table.add_row(label, f"{ratio['hits']}/{ratio['total']}", rate)
    table.add_row("  answerable but declined", str(metrics["false_refusals"]), "")
    table.add_row("  should have declined", str(metrics["missed_refusals"]), "")
    console.print(table)

    misses = [r for r in report.results if not r.refusal_correct or (r.answerable and not r.cites_expected)]
    if misses:
        console.print("\nMisses")
        for result in misses:
            if result.answerable and result.refused:
                why = f"declined (support {result.best_support:.2f})"
            elif not result.answerable:
                why = "answered, should have declined"
            else:
                why = "did not cite an expected source"
            rank = f", expected source at rank {result.expected_rank}" if result.expected_rank else ", expected source not retrieved"
            console.print(f"  {result.id}: {why}{rank if result.answerable else ''}")
