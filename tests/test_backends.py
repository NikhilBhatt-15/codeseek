from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from codeseek import embedder as emb
from codeseek.config import Settings
from codeseek.embedder import (
    BackendUnavailableError,
    GemmaEmbedder,
    OllamaEmbedder,
    make_embedder,
)
from codeseek.service import CodeSeek


class _FakeOllama(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []
    models: ClassVar[tuple[str, ...]] = ("embeddinggemma-2:latest",)

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, code: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, {"models": [{"name": n} for n in self.models]})

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(req)
        if req["model"] != "embeddinggemma-2":
            self._send(404, {"error": f"model '{req['model']}' not found"})
            return
        dim = req["dimensions"]
        # Unnormalised vectors so the client's normalisation is exercised.
        vecs = [[float(len(t) % 7 + 1)] * dim for t in req["input"]]
        self._send(200, {"model": req["model"], "embeddings": vecs})


@pytest.fixture
def ollama_url() -> Iterator[str]:
    _FakeOllama.requests = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_ollama_embedder_batches_prompts_and_normalises(ollama_url: str) -> None:
    s = Settings(ollama_url=ollama_url, batch_size=2, dim=128)
    e = OllamaEmbedder(s)
    seen: list[int] = []
    vecs = e.embed_documents([("a.py", "x"), ("b.py", "yy"), ("c.py", "zzz")], progress=seen.append)
    assert vecs.shape == (3, 128)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert seen == [2, 1]
    first = _FakeOllama.requests[0]
    assert first["input"][0] == "title: a.py | text: x"
    assert first["dimensions"] == 128
    assert first["options"]["num_thread"] >= 1

    q = e.embed_query("find x")
    assert q.shape == (128,)
    assert _FakeOllama.requests[-1]["input"] == ["task: code retrieval | query: find x"]
    assert e.model_id == "ollama:embeddinggemma-2"


def test_ollama_missing_model_hint(ollama_url: str) -> None:
    e = OllamaEmbedder(Settings(ollama_url=ollama_url, ollama_model="nope"))
    with pytest.raises(BackendUnavailableError, match="ollama pull nope"):
        e.embed_query("x")


def test_ollama_unreachable() -> None:
    e = OllamaEmbedder(Settings(ollama_url="http://127.0.0.1:9"))
    with pytest.raises(BackendUnavailableError, match="cannot reach Ollama"):
        e.embed_query("x")


def test_ollama_available(ollama_url: str) -> None:
    assert emb.ollama_available(Settings(ollama_url=ollama_url))
    assert not emb.ollama_available(Settings(ollama_url=ollama_url, ollama_model="other"))
    assert not emb.ollama_available(Settings(ollama_url="http://127.0.0.1:9"))


def test_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emb, "torch_available", lambda: True)
    monkeypatch.setattr(emb, "ollama_available", lambda s: True)
    auto = Settings()
    # New index: prefer Ollama when it is up.
    assert isinstance(make_embedder(auto), OllamaEmbedder)
    # Existing index: always reuse the backend that built it.
    assert isinstance(make_embedder(auto, "google/embeddinggemma-2"), GemmaEmbedder)
    assert isinstance(make_embedder(auto, "ollama:embeddinggemma-2"), OllamaEmbedder)
    # Explicit choice wins.
    assert isinstance(make_embedder(Settings(backend="torch")), GemmaEmbedder)

    monkeypatch.setattr(emb, "ollama_available", lambda s: False)
    assert isinstance(make_embedder(auto), GemmaEmbedder)

    monkeypatch.setattr(emb, "torch_available", lambda: False)
    with pytest.raises(BackendUnavailableError, match="no embedding backend"):
        make_embedder(auto)
    with pytest.raises(BackendUnavailableError, match=r"codeseek\[torch\]"):
        make_embedder(Settings(backend="torch"))


def test_invalid_backend_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(backend="gpu")


def test_status_needs_no_backend(
    repo: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(emb, "torch_available", lambda: False)
    monkeypatch.setattr(emb, "ollama_available", lambda s: False)
    with CodeSeek(repo, settings) as cs:
        assert not cs.status().indexed
        cs.clear()
        with pytest.raises(BackendUnavailableError):
            cs.index()
