"""Incremental indexing: only new or changed files are re-chunked and re-embedded."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from codeseek.chunking import Chunk, chunk_text
from codeseek.config import Settings
from codeseek.discovery import SourceFile, iter_source_files
from codeseek.embedder import Embedder
from codeseek.store import ChunkRow, IndexStore

log = logging.getLogger(__name__)

# Embed and commit in groups of roughly this many chunks, so an interrupted run
# keeps its progress and memory stays bounded on large repositories.
COMMIT_EVERY_CHUNKS = 256


@dataclass
class IndexReport:
    files_seen: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class _Pending:
    source: SourceFile
    sha256: str
    chunks: list[Chunk]
    is_new: bool


# Called with (chunks done, chunks total) after each embedding batch.
ProgressFn = Callable[[int, int], None]


def _read(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return raw.decode("utf-8-sig", errors="replace"), hashlib.sha256(raw).hexdigest()


class Indexer:
    def __init__(self, root: Path, store: IndexStore, embedder: Embedder, settings: Settings):
        self.root = root.resolve()
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def run(self, progress: ProgressFn | None = None) -> IndexReport:
        started = time.perf_counter()
        report = IndexReport()
        self.store.ensure_compatible(self.root, self.embedder.model_id, self.embedder.dim)

        existing = self.store.files()
        seen: set[str] = set()
        pending: list[_Pending] = []
        touched: list[SourceFile] = []

        for src in iter_source_files(self.root, self.settings.max_file_bytes):
            report.files_seen += 1
            seen.add(src.rel_path)
            old = existing.get(src.rel_path)
            if old and old.size == src.size and old.mtime_ns == src.mtime_ns:
                report.unchanged += 1
                continue
            try:
                text, digest = _read(src.path)
            except OSError as exc:
                report.errors.append(f"{src.rel_path}: {exc}")
                continue
            if old and old.sha256 == digest:
                touched.append(src)
                report.unchanged += 1
                continue
            chunks = chunk_text(
                text,
                max_lines=self.settings.chunk_max_lines,
                max_chars=self.settings.chunk_max_chars,
                overlap_lines=self.settings.chunk_overlap_lines,
            )
            pending.append(_Pending(src, digest, chunks, is_new=old is None))

        removed = [p for p in existing if p not in seen]
        with self.store.transaction() as conn:
            self.store.delete_files(conn, removed)
            for src in touched:
                self.store.touch_file(conn, src.rel_path, src.size, src.mtime_ns)
        report.removed = len(removed)

        total = sum(len(p.chunks) for p in pending)
        done = 0

        def on_batch(n: int) -> None:
            nonlocal done
            done = min(done + n, total)
            if progress:
                progress(done, total)

        for group in _group_by_chunk_count(pending, COMMIT_EVERY_CHUNKS):
            docs = [(p.source.rel_path, c.text) for p in group for c in p.chunks]
            vectors = self.embedder.embed_documents(docs, progress=on_batch)

            with self.store.transaction() as conn:
                offset = 0
                for p in group:
                    rows = [
                        ChunkRow(c.start_line, c.end_line, c.text, vectors[offset + i])
                        for i, c in enumerate(p.chunks)
                    ]
                    offset += len(p.chunks)
                    self.store.replace_file(
                        conn, p.source.rel_path, p.source.size, p.source.mtime_ns, p.sha256, rows
                    )
                    if p.is_new:
                        report.added += 1
                    else:
                        report.updated += 1
            report.chunks_embedded += len(docs)

        report.seconds = time.perf_counter() - started
        log.info("index complete: %s", report)
        return report


def _group_by_chunk_count(items: list[_Pending], limit: int) -> list[list[_Pending]]:
    groups: list[list[_Pending]] = []
    current: list[_Pending] = []
    count = 0
    for item in items:
        if current and count + len(item.chunks) > limit:
            groups.append(current)
            current, count = [], 0
        current.append(item)
        count += len(item.chunks)
    if current:
        groups.append(current)
    return groups
