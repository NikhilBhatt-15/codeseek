"""SQLite-backed index storage.

One database per repository, kept under the user cache directory so indexing
never writes into the repository itself. Vectors are stored as float32 blobs
and searched with an exact (brute-force) cosine scan, which is fast enough for
repositories up to a few hundred thousand chunks and needs no extra service.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from codeseek.embedder import Vectors

SCHEMA_VERSION = 1

Ids = npt.NDArray[np.int64]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id       INTEGER PRIMARY KEY,
    path     TEXT NOT NULL UNIQUE,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file_id ON chunks(file_id);
"""


class IndexMismatchError(RuntimeError):
    """The on-disk index was built with a different model, dimension or schema."""


@dataclass(frozen=True)
class FileRecord:
    id: int
    path: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class ChunkRow:
    start_line: int
    end_line: int
    text: str
    vector: Vectors


@dataclass(frozen=True)
class Hit:
    path: str
    start_line: int
    end_line: int
    score: float
    text: str


def index_dir_for(root: Path, index_root: Path) -> Path:
    """Stable per-repository directory: ``<name>-<hash of absolute path>``."""
    resolved = root.resolve()
    digest = hashlib.sha256(str(resolved).lower().encode()).hexdigest()[:12]
    return index_root / f"{resolved.name or 'root'}-{digest}"


class IndexStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._matrix: Vectors | None = None
        self._matrix_ids: Ids | None = None
        self._matrix_generation = -1

    @classmethod
    def for_repo(cls, root: Path, index_root: Path) -> IndexStore:
        return cls(index_dir_for(root, index_root) / "index.sqlite3")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IndexStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn
            self._bump_generation()

    # -- metadata ---------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _bump_generation(self) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('generation', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )

    def generation(self) -> int:
        return int(self.get_meta("generation") or 0)

    def ensure_compatible(self, root: Path, model_id: str, dim: int) -> None:
        """Stamp a fresh index, or verify an existing one matches this model and dimension."""
        stored_model = self.get_meta("model_id")
        if stored_model is None:
            with self._conn:
                self.set_meta("schema_version", str(SCHEMA_VERSION))
                self.set_meta("root", str(root.resolve()))
                self.set_meta("model_id", model_id)
                self.set_meta("dim", str(dim))
            return
        problems = []
        if self.get_meta("schema_version") != str(SCHEMA_VERSION):
            problems.append(f"schema {self.get_meta('schema_version')} != {SCHEMA_VERSION}")
        if stored_model != model_id:
            problems.append(f"model {stored_model!r} != {model_id!r}")
        if self.get_meta("dim") != str(dim):
            problems.append(f"dim {self.get_meta('dim')} != {dim}")
        if problems:
            raise IndexMismatchError(
                "existing index is incompatible (" + "; ".join(problems) + "). "
                "Rebuild it with `codeseek index --rebuild`."
            )

    # -- files and chunks -------------------------------------------------

    def files(self) -> dict[str, FileRecord]:
        rows = self._conn.execute("SELECT id, path, size, mtime_ns, sha256 FROM files")
        return {r[1]: FileRecord(*r) for r in rows}

    def delete_files(self, conn: sqlite3.Connection, paths: Iterable[str]) -> None:
        conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])

    def touch_file(self, conn: sqlite3.Connection, path: str, size: int, mtime_ns: int) -> None:
        """Update stat info for a file whose content hash did not change."""
        conn.execute(
            "UPDATE files SET size = ?, mtime_ns = ? WHERE path = ?", (size, mtime_ns, path)
        )

    def replace_file(
        self,
        conn: sqlite3.Connection,
        path: str,
        size: int,
        mtime_ns: int,
        sha256: str,
        chunks: Iterable[ChunkRow],
    ) -> None:
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        cur = conn.execute(
            "INSERT INTO files(path, size, mtime_ns, sha256) VALUES(?, ?, ?, ?)",
            (path, size, mtime_ns, sha256),
        )
        file_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO chunks(file_id, start_line, end_line, text, vector) VALUES(?, ?, ?, ?, ?)",
            [
                (file_id, c.start_line, c.end_line, c.text, c.vector.astype(np.float32).tobytes())
                for c in chunks
            ],
        )

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM files")
            self._conn.execute("DELETE FROM meta")
        self._matrix = None
        self._matrix_generation = -1

    def stats(self) -> dict[str, int]:
        files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"files": int(files), "chunks": int(chunks)}

    # -- search -----------------------------------------------------------

    def _load_matrix(self) -> tuple[Vectors, Ids]:
        gen = self.generation()
        if self._matrix is None or self._matrix_ids is None or gen != self._matrix_generation:
            dim = int(self.get_meta("dim") or 0)
            rows = self._conn.execute("SELECT id, vector FROM chunks ORDER BY id").fetchall()
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            matrix: Vectors
            if rows:
                flat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
                matrix = flat.reshape(len(rows), dim)
            else:
                matrix = np.zeros((0, dim), dtype=np.float32)
            self._matrix, self._matrix_ids, self._matrix_generation = matrix, ids, gen
        return self._matrix, self._matrix_ids

    def search(self, query_vec: Vectors, k: int, path_glob: str | None = None) -> list[Hit]:
        matrix, ids = self._load_matrix()
        if len(ids) == 0 or k <= 0:
            return []
        scores = matrix @ query_vec.astype(np.float32)

        if path_glob:
            from pathspec import GitIgnoreSpec

            spec = GitIgnoreSpec.from_lines([path_glob])
            allowed = {
                fid
                for fid, path in self._conn.execute("SELECT id, path FROM files")
                if spec.match_file(path)
            }
            chunk_file = dict(self._conn.execute("SELECT id, file_id FROM chunks").fetchall())
            mask = np.fromiter(
                (chunk_file.get(int(cid)) in allowed for cid in ids), dtype=bool, count=len(ids)
            )
            scores = np.where(mask, scores, -np.inf)

        n = min(k, int(np.isfinite(scores).sum()))
        if n == 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]

        hits: list[Hit] = []
        for i in top:
            row = self._conn.execute(
                "SELECT f.path, c.start_line, c.end_line, c.text FROM chunks c "
                "JOIN files f ON f.id = c.file_id WHERE c.id = ?",
                (int(ids[i]),),
            ).fetchone()
            if row is not None:
                hits.append(Hit(row[0], row[1], row[2], float(scores[i]), row[3]))
        return hits
