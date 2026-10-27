"""Optional MCP server exposing codeseek to coding agents.

Nothing registers this automatically. Install the ``mcp`` extra and add it to
an agent's config per project only when you want that agent to use it.

The server is scoped to one repository (``--root``, default: the working
directory) and only searches the existing index; it never builds one implicitly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from codeseek import __version__
from codeseek.embedder import BackendUnavailableError
from codeseek.service import CodeSeek, NotIndexedError
from codeseek.store import IndexMismatchError

MAX_RESULTS = 25


def build_server(cs: CodeSeek) -> object:
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "the MCP server needs the optional extra: pip install 'codeseek[mcp]'"
        ) from exc

    server = MCPServer("codeseek", version=__version__, log_level="WARNING")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def search_code(query: str, limit: int = 8, glob: str | None = None) -> str:
        """Semantic search over this repository's code by meaning, not exact text.

        Use for questions like "where is auth token refresh handled" when you do not
        know identifiers to grep for. Returns file:line ranges with snippets.

        Args:
            query: natural-language description or code fragment.
            limit: number of results (1-25).
            glob: optional gitignore-style filter, e.g. "src/**/*.py".
        """
        limit = max(1, min(limit, MAX_RESULTS))
        try:
            hits = cs.search(query, k=limit, path_glob=glob)
        except (NotIndexedError, IndexMismatchError, BackendUnavailableError, ValueError) as exc:
            return f"error: {exc}"
        if not hits:
            return "no results"
        return "\n\n".join(
            f"{h.path}:{h.start_line}-{h.end_line} (score {h.score:.3f})\n{h.text}" for h in hits
        )

    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codeseek-mcp", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository to serve")
    args = parser.parse_args(argv)
    # stdout carries the MCP protocol; logs must go to stderr.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    with CodeSeek(args.root) as cs:
        server = build_server(cs)
        server.run()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    sys.exit(main())
