import json
from pathlib import Path

import pytest

from codeseek import cli
from codeseek.config import Settings
from codeseek.embedder import HashEmbedder


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setenv("CODESEEK_INDEX_ROOT", str(settings.index_root))
    monkeypatch.setattr(
        "codeseek.service.make_embedder",
        lambda s, _index_model=None: HashEmbedder(dim=s.dim, model_id=s.model_id),
    )


def test_index_search_status_clear(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["index", str(repo), "--json"]) == cli.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["added"] == 3

    code = cli.main(["search", "refresh", "token", "-p", str(repo), "-k", "2", "--json"])
    assert code == cli.EXIT_OK
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["path"] == "src/auth.py"

    assert cli.main(["search", "pool", "-p", str(repo), "-l"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert all(":" in line for line in out.splitlines())

    assert cli.main(["status", str(repo)]) == cli.EXIT_OK
    assert "files:  3" in capsys.readouterr().out

    assert cli.main(["clear", str(repo)]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["status", str(repo)]) == cli.EXIT_INDEX


def test_search_unindexed_repo_exit_code(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["search", "x", "-p", str(repo)]) == cli.EXIT_INDEX
    assert "not indexed" in capsys.readouterr().err


def test_no_results_exit_code(repo: Path) -> None:
    cli.main(["index", str(repo), "-q"])
    assert cli.main(["search", "x", "-p", str(repo), "-g", "*.none"]) == cli.EXIT_NO_RESULTS


def test_bad_limit(repo: Path) -> None:
    assert cli.main(["search", "x", "-p", str(repo), "-k", "0"]) == cli.EXIT_USAGE


def test_shell_session(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["index", str(repo), "-q"])
    inputs = iter([":k 1", ":glob *.go", "database pool", ":glob", ":bogus", ":q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    assert cli.main(["shell", str(repo)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "src/db.go:" in out
    assert "(1 results in" in out
    assert "glob: (none)" in out


def test_shell_requires_index(repo: Path) -> None:
    assert cli.main(["shell", str(repo)]) == cli.EXIT_INDEX
