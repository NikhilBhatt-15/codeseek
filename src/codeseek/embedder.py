"""Embedding backends.

``GemmaEmbedder`` runs EmbeddingGemma 2 in-process with PyTorch, loading only the
text encoder (~270M parameters). ``OllamaEmbedder`` calls a local Ollama server
running the GGUF build of the same model. Both apply the model's retrieval
prompts: queries are prefixed with the code-retrieval task, documents are titled
with their file path.

The two backends produce close but not identical vectors (Q8 vs fp32 weights),
so each index records which backend built it and is only searched with that one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from codeseek.config import Settings

log = logging.getLogger(__name__)

Vectors = npt.NDArray[np.float32]

QUERY_PROMPT = "task: code retrieval | query: "


def format_document(rel_path: str, text: str) -> str:
    return f"title: {rel_path} | text: {text}"


def format_query(query: str) -> str:
    return f"{QUERY_PROMPT}{query}"


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(
        self,
        docs: Sequence[tuple[str, str]],
        progress: Callable[[int], None] | None = None,
    ) -> Vectors:
        """Embed ``(rel_path, text)`` pairs. Returns L2-normalised float32 rows."""
        ...

    def embed_query(self, query: str) -> Vectors:
        """Embed one query. Returns an L2-normalised float32 vector."""
        ...

    def warmup(self) -> None:
        """Load any heavy resources ahead of the first call."""
        ...


def _normalize(vectors: npt.NDArray[Any]) -> Vectors:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    out: Vectors = np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)
    return out


class GemmaEmbedder:
    """EmbeddingGemma 2, text encoder only. The model loads lazily on first use."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    @property
    def dim(self) -> int:
        return self._settings.dim

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._model = self._create_model()
        return self._model

    def _create_model(self) -> Any:
        import torch
        from sentence_transformers import SentenceTransformer

        device = self._settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # float16 produces NaNs / degraded vectors with this model; use bf16 or fp32 only.
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        log.info("loading %s on %s (%s)", self.model_id, device, dtype)

        kwargs: dict[str, Any] = {
            "device": device,
            "model_kwargs": {"dtype": dtype},
            # Skip the vision and audio encoders: code search is text-only.
            "config_kwargs": {"vision_config": None, "audio_config": None},
            "truncate_dim": self.dim,
        }
        try:
            # Use the cached copy without a network round-trip; works offline.
            model = SentenceTransformer(self.model_id, local_files_only=True, **kwargs)
        except OSError:
            log.info("%s not cached; downloading", self.model_id)
            model = SentenceTransformer(self.model_id, **kwargs)
        model.max_seq_length = self._settings.max_seq_length
        return model

    def warmup(self) -> None:
        """Load the model now instead of on first use."""
        self._load()

    def _encode(self, texts: list[str]) -> Vectors:
        model = self._load()
        out = model.encode(
            texts,
            batch_size=self._settings.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Re-normalise after Matryoshka truncation.
        return _normalize(out)

    def embed_documents(
        self,
        docs: Sequence[tuple[str, str]],
        progress: Callable[[int], None] | None = None,
    ) -> Vectors:
        if not docs:
            return np.zeros((0, self.dim), dtype=np.float32)
        texts = [format_document(path, text) for path, text in docs]
        # Sort by length so each batch pads to a similar size; restore order after.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result = np.empty((len(texts), self.dim), dtype=np.float32)
        step = self._settings.batch_size
        for start in range(0, len(order), step):
            idx = order[start : start + step]
            result[idx] = self._encode([texts[i] for i in idx])
            if progress:
                progress(len(idx))
        return result

    def embed_query(self, query: str) -> Vectors:
        vec: Vectors = self._encode([format_query(query)])[0]
        return vec


OLLAMA_PREFIX = "ollama:"


class BackendUnavailableError(RuntimeError):
    """The requested embedding backend cannot be used in this environment."""


class OllamaEmbedder:
    """EmbeddingGemma 2 served by a local Ollama instance (``ollama pull embeddinggemma-2``)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = settings.ollama_url.rstrip("/") + "/api/embed"
        threads = settings.ollama_threads or max(1, (os.cpu_count() or 4) * 3 // 4)
        self._options = {"num_thread": threads, "num_ctx": max(2048, settings.max_seq_length)}

    @property
    def model_id(self) -> str:
        return OLLAMA_PREFIX + self._settings.ollama_model

    @property
    def dim(self) -> int:
        return self._settings.dim

    def _embed(self, texts: list[str]) -> Vectors:
        body = {
            "model": self._settings.ollama_model,
            "input": texts,
            "dimensions": self.dim,
            "truncate": True,
            "keep_alive": "30m",
            "options": self._options,
        }
        req = urllib.request.Request(
            self._url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            hint = ""
            if exc.code == 404:
                hint = f" (run `ollama pull {self._settings.ollama_model}`)"
            raise BackendUnavailableError(f"Ollama error {exc.code}: {detail}{hint}") from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailableError(
                f"cannot reach Ollama at {self._settings.ollama_url}: {exc.reason}. "
                "Is `ollama serve` running?"
            ) from exc
        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
        if vectors.shape != (len(texts), self.dim):
            raise BackendUnavailableError(f"unexpected embedding shape {vectors.shape} from Ollama")
        return _normalize(vectors)

    def embed_documents(
        self,
        docs: Sequence[tuple[str, str]],
        progress: Callable[[int], None] | None = None,
    ) -> Vectors:
        if not docs:
            return np.zeros((0, self.dim), dtype=np.float32)
        texts = [format_document(path, text) for path, text in docs]
        step = self._settings.batch_size
        parts = []
        for start in range(0, len(texts), step):
            batch = texts[start : start + step]
            parts.append(self._embed(batch))
            if progress:
                progress(len(batch))
        return np.concatenate(parts)

    def embed_query(self, query: str) -> Vectors:
        vec: Vectors = self._embed([format_query(query)])[0]
        return vec

    def warmup(self) -> None:
        self._embed(["warmup"])


def ollama_available(settings: Settings) -> bool:
    """True if the Ollama server answers and has the configured model pulled."""
    url = settings.ollama_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            models = json.load(resp).get("models", [])
    except (OSError, ValueError):
        return False
    wanted = settings.ollama_model
    return any(m.get("name", "").split(":")[0] == wanted.split(":")[0] for m in models)


def torch_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def make_embedder(settings: Settings, index_model_id: str | None = None) -> Embedder:
    """Choose a backend: explicit setting, else the existing index's, else Ollama, else torch."""
    backend = settings.backend
    if backend == "auto" and index_model_id is not None:
        backend = "ollama" if index_model_id.startswith(OLLAMA_PREFIX) else "torch"
    if backend == "auto":
        if ollama_available(settings):
            backend = "ollama"
        elif torch_available():
            backend = "torch"
        else:
            raise BackendUnavailableError(
                "no embedding backend available. Either run Ollama with "
                f"`ollama pull {settings.ollama_model}`, or install the PyTorch backend: "
                "pip install 'codeseek[torch]'"
            )
    if backend == "ollama":
        return OllamaEmbedder(settings)
    if not torch_available():
        raise BackendUnavailableError(
            "the PyTorch backend is not installed: pip install 'codeseek[torch]'"
        )
    return GemmaEmbedder(settings)


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Builds a bag-of-tokens vector via feature hashing, so texts that share words
    land near each other. It is not semantic; it exists to test the pipeline.
    """

    def __init__(self, dim: int = 256, model_id: str = "hash-test") -> None:
        self._dim = dim
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> Vectors:
        v = np.zeros(self._dim, dtype=np.float32)
        for token in text.lower().replace("_", " ").split():
            token = "".join(ch for ch in token if ch.isalnum())
            if token:
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                v[h % self._dim] += 1.0
        return _normalize(v)

    def embed_documents(
        self,
        docs: Sequence[tuple[str, str]],
        progress: Callable[[int], None] | None = None,
    ) -> Vectors:
        rows = [self._vec(text) for _, text in docs]
        if progress:
            progress(len(rows))
        return np.stack(rows) if rows else np.zeros((0, self._dim), dtype=np.float32)

    def embed_query(self, query: str) -> Vectors:
        return self._vec(query)

    def warmup(self) -> None:
        pass
