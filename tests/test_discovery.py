from pathlib import Path

from codeseek.discovery import iter_source_files


def test_discovers_sources_and_skips_noise(repo: Path) -> None:
    found = {f.rel_path for f in iter_source_files(repo, max_file_bytes=1_000_000)}
    assert found == {"README.md", "src/auth.py", "src/db.go"}


def test_respects_max_file_size(repo: Path) -> None:
    found = {f.rel_path for f in iter_source_files(repo, max_file_bytes=100)}
    assert "src/auth.py" not in found
    assert "README.md" in found


def test_codeseekignore(repo: Path) -> None:
    (repo / ".codeseekignore").write_text("src/*.go\n")
    found = {f.rel_path for f in iter_source_files(repo, max_file_bytes=1_000_000)}
    assert "src/db.go" not in found
    assert "src/auth.py" in found


def test_ignored_directory_is_pruned(repo: Path) -> None:
    (repo / "generated").mkdir()
    (repo / "generated" / "api.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("generated/\n")
    found = {f.rel_path for f in iter_source_files(repo, max_file_bytes=1_000_000)}
    assert not any(p.startswith("generated/") for p in found)
