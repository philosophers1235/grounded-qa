"""Build the search index from a folder of documents, and persist it as JSON (+ .npy)."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, Chunk, chunk_documents
from .dense import DenseIndex, Embedder
from .documents import load_directory, load_document
from .text import tokenize

FORMAT_VERSION = 1


class IndexNotFound(FileNotFoundError):
    pass


class Index:
    """Chunks plus the structures needed to search them.

    Only the chunks (and dense vectors, if any) are stored. BM25 statistics are
    rebuilt on load: it takes milliseconds and keeps the on-disk format trivial.
    """

    def __init__(self, chunks: list[Chunk], manifest: dict, vectors: np.ndarray | None = None):
        self.chunks = chunks
        self.manifest = manifest
        self.vectors = vectors
        self.tokens = [tokenize(chunk.search_text) for chunk in chunks]
        self.token_sets = [set(tokens) for tokens in self.tokens]
        self.bm25 = BM25(self.tokens)
        self.dense = DenseIndex(vectors) if vectors is not None else None

    @property
    def embedder_spec(self) -> str | None:
        return self.manifest.get("embedder")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        (directory / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        vectors_path = directory / "vectors.npy"
        if self.vectors is not None:
            np.save(vectors_path, self.vectors)
        elif vectors_path.exists():
            vectors_path.unlink()

    @classmethod
    def load(cls, directory: Path) -> "Index":
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise IndexNotFound(f"no index in {directory}; run `gqa index <folder>` first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"index in {directory} has format {manifest.get('format')}, expected {FORMAT_VERSION}; re-run `gqa index`")
        chunks = [Chunk.from_dict(c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        vectors_path = directory / "vectors.npy"
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        return cls(chunks, manifest, vectors)


def build_index(
    source: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    embedder: Embedder | None = None,
    embedder_spec: str | None = None,
) -> Index:
    documents = load_directory(source) if source.is_dir() else [load_document(source)]
    if not documents:
        raise ValueError(f"no .md, .txt or .pdf files found in {source}")

    chunks = chunk_documents(documents, max_chars, overlap_chars)
    vectors = None
    if embedder is not None:
        vectors = np.asarray(embedder.embed([chunk.search_text for chunk in chunks]), dtype=np.float32)

    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk.doc_id] = chunk_counts.get(chunk.doc_id, 0) + 1

    manifest = {
        "format": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "chunking": {"max_chars": max_chars, "overlap_chars": overlap_chars},
        "embedder": embedder_spec,
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "path": doc.path,
                "sha256": hashlib.sha256(Path(doc.path).read_bytes()).hexdigest(),
                "chunks": chunk_counts.get(doc.doc_id, 0),
            }
            for doc in documents
        ],
    }
    return Index(chunks, manifest, vectors)
