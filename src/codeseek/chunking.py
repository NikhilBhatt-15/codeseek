"""Split source files into chunks that line up with code structure where possible.

The chunker is language-agnostic. It treats a top-level declaration (a line at
indentation zero that starts a definition, or any unindented line after a blank
line) as a natural boundary, packs the segments between boundaries into chunks
up to a size budget, and falls back to overlapping line windows for segments
that are too big on their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DECL_RE = re.compile(
    r"^(?:export\s+|pub(?:\([^)]*\))?\s+|public\s+|private\s+|protected\s+|static\s+|"
    r"async\s+|default\s+|abstract\s+|final\s+|internal\s+|override\s+)*"
    r"(?:def|class|func|fn|function|interface|struct|enum|trait|impl|type|module|"
    r"namespace|object|record|macro_rules!|const|let|var|val|@\w+|#\[|##?\s)"
)
_CLOSING_RE = re.compile(r"^[\]\)\}][\]\)\};,]*\s*$")
# A declaration directly under one of these (decorator, attribute, comment) stays attached.
_GLUE_PREFIXES = ("@", "#", "//", "/*", "--")


@dataclass(frozen=True)
class Chunk:
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str


def _is_boundary(lines: list[str], i: int) -> bool:
    line = lines[i]
    if not line.strip() or line[0].isspace() or _CLOSING_RE.match(line):
        return False
    if _DECL_RE.match(line):
        # Keep decorators/attributes/comments glued to what follows them.
        prev = lines[i - 1] if i > 0 else ""
        return not (prev and not prev[0].isspace() and prev.startswith(_GLUE_PREFIXES))
    return i > 0 and not lines[i - 1].strip()


def _segments(lines: list[str]) -> list[tuple[int, int]]:
    """Return half-open [start, end) line ranges split at top-level boundaries."""
    starts = [0, *(i for i in range(1, len(lines)) if _is_boundary(lines, i))]
    ends = [*starts[1:], len(lines)]
    return list(zip(starts, ends, strict=True))


def _windows(start: int, end: int, max_lines: int, overlap: int) -> list[tuple[int, int]]:
    step = max_lines - overlap
    out: list[tuple[int, int]] = []
    s = start
    while s < end:
        e = min(s + max_lines, end)
        out.append((s, e))
        if e == end:
            break
        s += step
    return out


def _trim_blank(lines: list[str], start: int, end: int) -> tuple[int, int]:
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start, end


def chunk_text(text: str, max_lines: int, max_chars: int, overlap_lines: int) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    def size(s: int, e: int) -> int:
        return sum(len(line) + 1 for line in lines[s:e])

    # 1) Split oversized segments into windows so every piece fits the budget.
    pieces: list[tuple[int, int]] = []
    for s, e in _segments(lines):
        s, e = _trim_blank(lines, s, e)
        if s >= e:
            continue
        if e - s <= max_lines and size(s, e) <= max_chars:
            pieces.append((s, e))
            continue
        for ws, we in _windows(s, e, max_lines, overlap_lines):
            # A window can still exceed max_chars with very long lines; shrink it.
            while we - ws > 1 and size(ws, we) > max_chars:
                we -= max(1, (we - ws) // 4)
            pieces.append((ws, we))

    # 2) Greedily merge neighbouring small pieces (and the blank lines between them)
    #    up to the budget. Overlapping windows are never merged.
    merged: list[tuple[int, int]] = []
    for s, e in pieces:
        if merged:
            ms, me = merged[-1]
            if s >= me and e - ms <= max_lines and size(ms, e) <= max_chars:
                merged[-1] = (ms, e)
                continue
        merged.append((s, e))

    chunks: list[Chunk] = []
    seen: set[tuple[int, int]] = set()
    for s, e in merged:
        s, e = _trim_blank(lines, s, e)
        if s >= e or (s, e) in seen:
            continue
        seen.add((s, e))
        body = "\n".join(lines[s:e])
        if len(body) > max_chars:  # a single enormous line (e.g. minified code)
            body = body[:max_chars]
        chunks.append(Chunk(start_line=s + 1, end_line=e, text=body))
    return chunks
