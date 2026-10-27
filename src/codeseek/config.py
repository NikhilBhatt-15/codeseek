"""Runtime settings.

Every setting has a sane default and can be overridden with a ``CODESEEK_*``
environment variable, so the CLI and the MCP server behave identically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from platformdirs import user_cache_dir

DEFAULT_MODEL = "google/embeddinggemma-2"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "embeddinggemma-2"
BACKENDS = ("auto", "torch", "ollama")

# Matryoshka dimensions EmbeddingGemma 2 is trained to support.
SUPPORTED_DIMS = (768, 512, 256, 128)


def _default_index_root() -> Path:
    return Path(user_cache_dir("codeseek", appauthor=False)) / "indexes"


@dataclass(frozen=True)
class Settings:
    # "auto" reuses the backend an existing index was built with; for a new index it
    # picks Ollama when the server is reachable, otherwise local PyTorch.
    backend: str = "auto"
    model_id: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    # 0 = auto (3/4 of logical CPUs). Ollama's own default leaves CPU cores idle.
    ollama_threads: int = 0
    # 256 is close to lossless for retrieval and 3x smaller than 768.
    dim: int = 256
    batch_size: int = 16
    # Upper bound on tokens fed to the model per chunk. Code averages ~3 chars/token,
    # so the chunk budget below fits; throughput on CPU drops sharply past ~512.
    max_seq_length: int = 512
    # Chunking.
    chunk_max_lines: int = 50
    chunk_max_chars: int = 1500
    chunk_overlap_lines: int = 5
    # Discovery.
    max_file_bytes: int = 1_000_000
    index_root: Path = field(default_factory=_default_index_root)
    device: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {self.dim}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.chunk_overlap_lines >= self.chunk_max_lines:
            raise ValueError("chunk_overlap_lines must be smaller than chunk_max_lines")

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from defaults, then ``CODESEEK_*`` env vars, then explicit overrides."""
        base = cls()
        env: dict[str, object] = {}
        for f in fields(cls):
            raw = os.environ.get(f"CODESEEK_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            current = getattr(base, f.name)
            if isinstance(current, bool):
                env[f.name] = raw.lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                env[f.name] = int(raw)
            elif isinstance(current, Path):
                env[f.name] = Path(raw).expanduser()
            else:
                env[f.name] = raw
        env.update({k: v for k, v in overrides.items() if v is not None})
        return replace(base, **env)  # type: ignore[arg-type]
