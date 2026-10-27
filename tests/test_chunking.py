from codeseek.chunking import chunk_text


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("", 20, 2000, 2) == []
    assert chunk_text("\n\n  \n", 20, 2000, 2) == []


def test_small_file_is_one_chunk() -> None:
    text = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    chunks = chunk_text(text, max_lines=20, max_chars=2000, overlap_lines=2)
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 6)


def test_splits_at_top_level_definitions() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(8))
    text = f"def first():\n{body}\n\n\ndef second():\n{body}\n"
    chunks = chunk_text(text, max_lines=10, max_chars=2000, overlap_lines=2)
    assert [c.text.splitlines()[0] for c in chunks] == ["def first():", "def second():"]


def test_decorator_stays_with_function() -> None:
    body = "\n".join(f"    y{i} = {i}" for i in range(8))
    text = f"x = 1\n\n@cache\ndef f():\n{body}\n"
    chunks = chunk_text(text, max_lines=10, max_chars=2000, overlap_lines=2)
    assert any(c.text.startswith("@cache\ndef f():") for c in chunks)


def test_oversized_segment_uses_overlapping_windows() -> None:
    text = "def big():\n" + "\n".join(f"    v{i} = {i}" for i in range(50))
    chunks = chunk_text(text, max_lines=20, max_chars=10_000, overlap_lines=5)
    assert len(chunks) >= 3
    assert all(c.end_line - c.start_line + 1 <= 20 for c in chunks)
    # consecutive windows overlap by the configured amount
    assert chunks[1].start_line == chunks[0].end_line - 5 + 1
    # every line is covered
    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered == set(range(1, 52))


def test_respects_char_budget_with_long_lines() -> None:
    text = "\n".join("z" * 500 for _ in range(30))
    chunks = chunk_text(text, max_lines=20, max_chars=2000, overlap_lines=2)
    assert all(len(c.text) <= 2000 for c in chunks)


def test_line_numbers_match_text() -> None:
    lines = [f"line_{i}" if i % 7 else "" for i in range(1, 120)]
    text = "\n".join(lines)
    for c in chunk_text(text, max_lines=15, max_chars=5000, overlap_lines=3):
        assert c.text.splitlines() == lines[c.start_line - 1 : c.end_line]
