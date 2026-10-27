"""Command-line interface.

Exit codes: 0 success, 1 no results, 2 usage error, 3 index missing or incompatible,
4 embedding backend unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from codeseek import __version__
from codeseek.config import BACKENDS, SUPPORTED_DIMS, Settings
from codeseek.embedder import BackendUnavailableError
from codeseek.indexer import ProgressFn
from codeseek.service import CodeSeek, NotIndexedError
from codeseek.store import Hit, IndexMismatchError

EXIT_OK, EXIT_NO_RESULTS, EXIT_USAGE, EXIT_INDEX, EXIT_BACKEND = 0, 1, 2, 3, 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeseek",
        description="Local semantic code search powered by EmbeddingGemma 2.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv)")
    parser.add_argument(
        "--dim", type=int, choices=SUPPORTED_DIMS, help="embedding size (default 256)"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help="embedding backend (default auto: the index's own, else Ollama, else PyTorch)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build or update the index for a repository")
    p_index.add_argument("path", nargs="?", default=".", type=Path)
    p_index.add_argument("--rebuild", action="store_true", help="discard and rebuild the index")
    p_index.add_argument("--json", action="store_true", help="print the report as JSON")
    p_index.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    p_search = sub.add_parser("search", help="search an indexed repository")
    p_search.add_argument("query", nargs="+", help="natural-language or code query")
    p_search.add_argument("-p", "--path", default=".", type=Path, help="repository (default .)")
    p_search.add_argument("-k", "--limit", type=int, default=10, help="results (default 10)")
    p_search.add_argument("-g", "--glob", help="only files matching this gitignore-style glob")
    p_search.add_argument("--update", action="store_true", help="refresh the index first")
    p_search.add_argument("--json", action="store_true", help="print results as JSON")
    p_search.add_argument("-l", "--files-only", action="store_true", help="print locations only")
    p_search.add_argument(
        "--lines", type=int, default=8, help="snippet lines per result (default 8, 0 = none)"
    )

    p_shell = sub.add_parser(
        "shell", help="interactive search; loads the model once for fast repeated queries"
    )
    p_shell.add_argument("path", nargs="?", default=".", type=Path)
    p_shell.add_argument("-k", "--limit", type=int, default=5, help="results (default 5)")
    p_shell.add_argument("--lines", type=int, default=6, help="snippet lines (default 6)")

    p_status = sub.add_parser("status", help="show index statistics")
    p_status.add_argument("path", nargs="?", default=".", type=Path)
    p_status.add_argument("--json", action="store_true")

    p_clear = sub.add_parser("clear", help="delete the index for a repository")
    p_clear.add_argument("path", nargs="?", default=".", type=Path)
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    if verbosity < 2:
        # The ML stack is chatty; keep it quiet unless debugging.
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        for name in ("sentence_transformers", "transformers", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.ERROR)


def _color(enabled: bool) -> dict[str, str]:
    if not enabled:
        return {"path": "", "dim": "", "score": "", "reset": ""}
    return {"path": "\033[35m", "dim": "\033[2m", "score": "\033[32m", "reset": "\033[0m"}


def _print_hits(hits: list[Hit], lines: int, files_only: bool) -> None:
    c = _color(sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    for hit in hits:
        loc = f"{hit.path}:{hit.start_line}-{hit.end_line}"
        if files_only:
            print(loc)
            continue
        print(f"{c['path']}{loc}{c['reset']}  {c['score']}{hit.score:.3f}{c['reset']}")
        if lines > 0:
            snippet = hit.text.splitlines()[:lines]
            for offset, line in enumerate(snippet):
                print(f"{c['dim']}{hit.start_line + offset:>6} |{c['reset']} {line}")
            print()


def _progress_printer(quiet: bool) -> ProgressFn | None:
    if quiet or not sys.stderr.isatty():
        return None

    def progress(done: int, total: int) -> None:
        pct = 100 * done // total if total else 100
        print(f"\rembedding {done}/{total} chunks ({pct}%)", end="", file=sys.stderr, flush=True)
        if done >= total:
            print(file=sys.stderr)

    return progress


def _cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    with CodeSeek(args.path, settings) as cs:
        report = cs.index(rebuild=args.rebuild, progress=_progress_printer(args.quiet))
        if args.json:
            print(json.dumps({"root": str(cs.root), **asdict(report)}, indent=2))
        elif not args.quiet:
            print(
                f"{cs.root}: {report.files_seen} files "
                f"(+{report.added} ~{report.updated} -{report.removed} ={report.unchanged}), "
                f"{report.chunks_embedded} chunks embedded in {report.seconds:.1f}s"
            )
        for err in report.errors:
            print(f"warning: {err}", file=sys.stderr)
    return EXIT_OK


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    if args.limit < 1:
        print("error: --limit must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    query = " ".join(args.query)
    with CodeSeek(args.path, settings) as cs:
        if args.update:
            cs.index(progress=_progress_printer(quiet=False))
        hits = cs.search(query, k=args.limit, path_glob=args.glob)
    if args.json:
        print(json.dumps([asdict(h) for h in hits], indent=2))
    else:
        _print_hits(hits, args.lines, args.files_only)
    return EXIT_OK if hits else EXIT_NO_RESULTS


_SHELL_HELP = """\
type a query to search. commands:
  :k N        results per query          :glob PAT   only matching files (:glob to reset)
  :reindex    update the index           :q          quit"""


def _quiet_warmup(cs: CodeSeek) -> None:
    # Failures resurface with a proper message on the first real query.
    try:
        cs.embedder.warmup()
    except Exception:
        logging.getLogger(__name__).debug("warmup failed", exc_info=True)


def _cmd_shell(args: argparse.Namespace, settings: Settings) -> int:
    limit, glob = args.limit, None
    with CodeSeek(args.path, settings) as cs:
        if not cs.status().indexed:
            raise NotIndexedError(
                f"{cs.root} is not indexed yet. Run `codeseek index` in it first."
            )
        # Load the model while the user types the first query.
        threading.Thread(target=_quiet_warmup, args=(cs,), daemon=True).start()
        print(f"codeseek shell: {cs.root}\n{_SHELL_HELP}")
        while True:
            try:
                line = input("codeseek> ").strip()
            except EOFError:
                print()
                return EXIT_OK
            if not line:
                continue
            if line in {":q", ":quit", ":exit"}:
                return EXIT_OK
            if line.startswith(":k"):
                try:
                    limit = max(1, int(line[2:].strip()))
                except ValueError:
                    print("usage: :k N")
                continue
            if line.startswith(":glob"):
                glob = line[5:].strip() or None
                print(f"glob: {glob or '(none)'}")
                continue
            if line == ":reindex":
                report = cs.index(progress=_progress_printer(quiet=False))
                print(f"+{report.added} ~{report.updated} -{report.removed}")
                continue
            if line.startswith(":"):
                print(_SHELL_HELP)
                continue
            started = time.perf_counter()
            hits = cs.search(line, k=limit, path_glob=glob)
            _print_hits(hits, args.lines, files_only=False)
            print(f"({len(hits)} results in {time.perf_counter() - started:.2f}s)")


def _cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    with CodeSeek(args.path, settings) as cs:
        st = cs.status()
    if args.json:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(st).items()}))
        return EXIT_OK
    print(f"root:   {st.root}")
    print(f"index:  {st.index_path}")
    if not st.indexed:
        print("state:  not indexed")
        return EXIT_INDEX
    print(f"model:  {st.model_id} ({st.dim}d)")
    print(f"files:  {st.files}")
    print(f"chunks: {st.chunks}")
    return EXIT_OK


def _cmd_clear(args: argparse.Namespace, settings: Settings) -> int:
    with CodeSeek(args.path, settings) as cs:
        cs.clear()
        print(f"cleared index for {cs.root}")
    return EXIT_OK


_COMMANDS = {
    "index": _cmd_index,
    "search": _cmd_search,
    "shell": _cmd_shell,
    "status": _cmd_status,
    "clear": _cmd_clear,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        settings = Settings.from_env(dim=args.dim, backend=args.backend)
        return _COMMANDS[args.command](args, settings)
    except (NotIndexedError, IndexMismatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INDEX
    except BackendUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BACKEND
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\ninterrupted (progress so far is saved)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
