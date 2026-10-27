from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeseek.config import Settings
from codeseek.embedder import HashEmbedder
from codeseek.service import CodeSeek

AUTH_PY = '''\
import time


def refresh_access_token(session):
    """Refresh the OAuth access token when it is about to expire."""
    if session.expires_at - time.time() < 60:
        session.token = session.client.refresh(session.refresh_token)
    return session.token


class LoginForm:
    def validate_password(self, password):
        return len(password) >= 12
'''

DB_GO = """\
package db

import "database/sql"

// OpenPool opens a connection pool to the postgres database.
func OpenPool(dsn string) (*sql.DB, error) {
\treturn sql.Open("postgres", dsn)
}
"""

README = "# demo\n\nA tiny demo repository used by the codeseek test suite.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src" / "auth.py").write_text(AUTH_PY, encoding="utf-8")
    (root / "src" / "db.go").write_text(DB_GO, encoding="utf-8")
    (root / "README.md").write_text(README, encoding="utf-8")
    (root / "node_modules" / "lib").mkdir(parents=True)
    (root / "node_modules" / "lib" / "index.js").write_text("module.exports = 1\n")
    (root / "secret.log").write_text("ignored by gitignore\n")
    (root / ".gitignore").write_text("*.log\n")
    (root / "image.py").write_bytes(b"\x00\x01binary")
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(index_root=tmp_path / "indexes", chunk_max_lines=20, chunk_overlap_lines=2)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)


@pytest.fixture
def cs(repo: Path, settings: Settings, embedder: HashEmbedder) -> Iterator[CodeSeek]:
    with CodeSeek(repo, settings, embedder) as instance:
        yield instance
