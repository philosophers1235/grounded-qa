"""Terminal output for answers."""

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .answers import MODE_CLAUDE, Answer


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
