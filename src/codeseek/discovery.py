"""Find the source files in a repository that are worth indexing."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

# Directories that are never source, regardless of .gitignore.
ALWAYS_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
        "dist", "build", "target", "out", ".next", ".nuxt", ".gradle", "vendor",
        ".terraform", "coverage", ".codeseek",
    }
)  # fmt: skip

# Extensions we treat as source or prose worth searching.
TEXT_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs",
        ".java", ".kt", ".kts", ".scala", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
        ".hh", ".cs", ".fs", ".swift", ".m", ".mm", ".rb", ".php", ".lua", ".dart",
        ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".r", ".jl", ".zig", ".nim",
        ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".sql", ".graphql", ".proto",
        ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".astro",
        ".md", ".mdx", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini",
        ".cfg", ".xml", ".gradle", ".cmake", ".tf", ".nix",
    }
)  # fmt: skip

TEXT_FILENAMES = frozenset(
    {"Dockerfile", "Makefile", "CMakeLists.txt", "Justfile", "Rakefile", "Gemfile"}
)

# Generated files that match TEXT_EXTENSIONS but only add noise.
SKIP_FILENAMES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
        "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum",
    }
)  # fmt: skip

_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel_path: str  # POSIX-style, relative to the repo root
    size: int
    mtime_ns: int


def _load_ignore_spec(root: Path) -> pathspec.GitIgnoreSpec:
    lines: list[str] = []
    for name in (".gitignore", ".codeseekignore"):
        ignore_file = root / name
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8", errors="replace").splitlines())
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _is_candidate(name: str) -> bool:
    if name in SKIP_FILENAMES or name.endswith(".min.js") or name.endswith(".min.css"):
        return False
    return name in TEXT_FILENAMES or Path(name).suffix.lower() in TEXT_EXTENSIONS


def looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True


def iter_source_files(root: Path, max_file_bytes: int) -> Iterator[SourceFile]:
    """Yield indexable files under ``root``, honouring .gitignore and .codeseekignore.

    Only the ignore files at the repo root are read; nested ignore files are not.
    """
    root = root.resolve()
    spec = _load_ignore_spec(root)

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_dir = current.relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else f"{rel_dir}/"

        # Prune in place so os.walk never descends into skipped directories.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in ALWAYS_SKIP_DIRS
            and not d.startswith(".")
            and not spec.match_file(f"{prefix}{d}/")
        )

        for name in sorted(filenames):
            rel = f"{prefix}{name}"
            if not _is_candidate(name) or spec.match_file(rel):
                continue
            path = current / name
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0 or stat.st_size > max_file_bytes or looks_binary(path):
                continue
            yield SourceFile(path=path, rel_path=rel, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
