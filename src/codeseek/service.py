"""High-level operations shared by the CLI and the MCP server."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from codeseek.config import Settings
from codeseek.embedder import Embedder, make_embedder
from codeseek.indexer import Indexer, IndexReport, ProgressFn
from codeseek.store import Hit, IndexStore


class NotIndexedError(RuntimeError):
    """Search was requested for a repository that has no index yet."""


def find_repo_root(start: Path) -> Path:
    """Return the nearest ancestor containing ``.git``, or ``start`` itself."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


@dataclass
class Status:
    root: Path
    index_path: Path
    indexed: bool
    files: int
    chunks: int
    model_id: str | None
    dim: int | None


class CodeSeek:
    """A repository plus its index. Reuse one instance to keep the model loaded."""

    def __init__(
        self, root: Path, settings: Settings | None = None, embedder: Embedder | None = None
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.root = find_repo_root(root)
        self.store = IndexStore.for_repo(self.root, self.settings.index_root)
        self._embedder = embedder
        self._embedder_lock = threading.Lock()

    @property
    def embedder(self) -> Embedder:
        """Created on first use, so status/clear work without any backend installed."""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = make_embedder(self.settings, self.store.get_meta("model_id"))
        return self._embedder

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> CodeSeek:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def index(self, rebuild: bool = False, progress: ProgressFn | None = None) -> IndexReport:
        if rebuild:
            self.store.clear()
        return Indexer(self.root, self.store, self.embedder, self.settings).run(progress)

    def search(self, query: str, k: int = 10, path_glob: str | None = None) -> list[Hit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if self.store.get_meta("model_id") is None:
            raise NotIndexedError(
                f"{self.root} is not indexed yet. Run `codeseek index` in it first."
            )
        self.store.ensure_compatible(self.root, self.embedder.model_id, self.embedder.dim)
        return self.store.search(self.embedder.embed_query(query), k, path_glob)

    def status(self) -> Status:
        stats = self.store.stats()
        dim = self.store.get_meta("dim")
        return Status(
            root=self.root,
            index_path=self.store.db_path,
            indexed=self.store.get_meta("model_id") is not None,
            files=stats["files"],
            chunks=stats["chunks"],
            model_id=self.store.get_meta("model_id"),
            dim=int(dim) if dim else None,
        )

    def clear(self) -> None:
        self.store.clear()
