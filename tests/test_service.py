from pathlib import Path

import pytest

from codeseek.config import Settings
from codeseek.embedder import HashEmbedder, format_document, format_query
from codeseek.service import CodeSeek, NotIndexedError, find_repo_root
from codeseek.store import IndexMismatchError, index_dir_for


def test_index_then_search(cs: CodeSeek) -> None:
    report = cs.index()
    assert (report.added, report.updated, report.removed) == (3, 0, 0)
    assert report.chunks_embedded > 0

    hits = cs.search("refresh access token session", k=3)
    assert hits[0].path == "src/auth.py"
    assert hits[0].score >= hits[-1].score

    hits = cs.search("postgres database connection pool", k=1)
    assert hits[0].path == "src/db.go"


def test_reindex_is_incremental(cs: CodeSeek, repo: Path) -> None:
    cs.index()
    again = cs.index()
    assert (again.added, again.updated, again.removed, again.chunks_embedded) == (0, 0, 0, 0)
    assert again.unchanged == 3

    (repo / "src" / "db.go").write_text("package db\n\nfunc Close() {}\n")
    (repo / "README.md").unlink()
    (repo / "src" / "new.py").write_text("def brand_new_helper():\n    pass\n")
    report = cs.index()
    assert (report.added, report.updated, report.removed) == (1, 1, 1)
    assert cs.status().files == 3
    assert cs.search("brand new helper", k=1)[0].path == "src/new.py"


def test_touch_without_content_change_skips_embedding(cs: CodeSeek, repo: Path) -> None:
    cs.index()
    target = repo / "src" / "auth.py"
    target.write_text(target.read_text())  # new mtime, same bytes
    report = cs.index()
    assert report.chunks_embedded == 0
    assert report.unchanged == 3


def test_glob_filter(cs: CodeSeek) -> None:
    cs.index()
    hits = cs.search("database", k=10, path_glob="*.py")
    assert hits and all(h.path.endswith(".py") for h in hits)
    assert cs.search("database", k=10, path_glob="*.nothing") == []


def test_search_before_index_raises(cs: CodeSeek) -> None:
    with pytest.raises(NotIndexedError):
        cs.search("anything")


def test_empty_query_rejected(cs: CodeSeek) -> None:
    cs.index()
    with pytest.raises(ValueError):
        cs.search("   ")


def test_model_mismatch_detected(repo: Path, settings: Settings) -> None:
    with CodeSeek(repo, settings, HashEmbedder(dim=256, model_id="a")) as cs:
        cs.index()
    with (
        CodeSeek(repo, settings, HashEmbedder(dim=256, model_id="b")) as cs,
        pytest.raises(IndexMismatchError),
    ):
        cs.search("token")


def test_rebuild_clears_index(cs: CodeSeek) -> None:
    cs.index()
    report = cs.index(rebuild=True)
    assert report.added == 3


def test_index_lives_outside_repo(cs: CodeSeek, repo: Path, settings: Settings) -> None:
    cs.index()
    assert cs.store.db_path.parent == index_dir_for(repo, settings.index_root)
    assert not cs.store.db_path.is_relative_to(repo)


def test_find_repo_root_walks_up(repo: Path) -> None:
    assert find_repo_root(repo / "src") == repo.resolve()


def test_prompt_formats() -> None:
    assert format_query("x") == "task: code retrieval | query: x"
    assert format_document("a/b.py", "code") == "title: a/b.py | text: code"


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        Settings(dim=300)
    with pytest.raises(ValueError):
        Settings(chunk_max_lines=5, chunk_overlap_lines=5)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODESEEK_DIM", "512")
    monkeypatch.setenv("CODESEEK_BATCH_SIZE", "4")
    s = Settings.from_env()
    assert (s.dim, s.batch_size) == (512, 4)
    assert Settings.from_env(dim=128).dim == 128
