import argparse
import dataclasses
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS
from .claude import ClaudeError
from .config import Settings
from .dense import load_embedder
from .index import IndexNotFound, build_index
from .pipeline import QAService
from .render import print_answer

console = Console()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.index_dir:
        settings = dataclasses.replace(settings, index_dir=args.index_dir)
    if getattr(args, "top_k", None):
        settings = dataclasses.replace(settings, top_k=args.top_k)

    try:
        return args.handler(args, settings)
    except IndexNotFound as exc:
        console.print(f"[red]{exc}[/red]")
    except ClaudeError as exc:
        console.print(f"[red]{exc}[/red]")
    return 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-dir", type=Path, help="where the index lives (default: $GQA_INDEX_DIR or .gqa_index)")

    parser = argparse.ArgumentParser(prog="gqa", description="Answer questions from your own documents, with sources.")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", parents=[common], help="build the index from a folder or a single file")
    index.add_argument("source", type=Path)
    index.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="soft chunk size limit")
    index.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CHARS, help="overlap between chunks, in characters")
    index.add_argument("--embedder", help="optional dense embedder, as package.module:factory (default: $GQA_EMBEDDER)")
    index.set_defaults(handler=cmd_index)

    ask = commands.add_parser("ask", parents=[common], help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--offline", action="store_true", help="answer without calling Claude")
    ask.add_argument("-k", "--top-k", type=int, help="number of chunks to retrieve")
    ask.add_argument("--show-retrieval", action="store_true", help="print the retrieved chunks and their scores")
    ask.set_defaults(handler=cmd_ask)

    return parser


def cmd_index(args, settings: Settings) -> int:
    if not args.source.exists():
        console.print(f"[red]{args.source} does not exist[/red]")
        return 1
    spec = args.embedder or settings.embedder
    embedder = load_embedder(spec) if spec else None
    index = build_index(args.source, args.max_chars, args.overlap, embedder=embedder, embedder_spec=spec)
    index.save(settings.index_dir)

    for doc in index.manifest["documents"]:
        console.print(f"  {doc['chunks']:>3} chunks  {doc['path']}")
    dense = f", dense vectors from {spec}" if spec else ""
    console.print(f"Indexed {len(index.manifest['documents'])} documents, {len(index.chunks)} chunks{dense} -> {settings.index_dir}")
    return 0


def cmd_ask(args, settings: Settings) -> int:
    service = QAService.from_settings(settings, offline=args.offline)
    answer = service.ask(args.question)
    print_answer(console, answer, show_retrieval=args.show_retrieval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
