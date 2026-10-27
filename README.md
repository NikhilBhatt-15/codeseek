# codeseek

Search your codebase by meaning, not by exact text, entirely on your own machine.

```
$ codeseek search "where do we refresh expired auth tokens"
src/auth/session.py:41-78  0.812
    41 | def refresh_access_token(session):
    42 |     """Refresh the OAuth access token when it is about to expire."""
    ...
```

codeseek is built on [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2),
Google DeepMind's open on-device embedding model (Apache 2.0). It uses only the model's
**270M-parameter text encoder** and runs it through [Ollama](https://ollama.com) or
in-process with PyTorch. No code, query or index ever leaves your computer.

## Features

- **Semantic search.** `"keep users signed in when credentials expire"` finds
  `refresh_access_token` even though the two share no words.
- **Local and offline.** No API keys and no telemetry. After the one-time model
  download it works with no network.
- **Incremental indexing.** Re-runs only re-embed files whose content changed. A file
  that was saved without changes is detected by hash and skipped.
- **Stays out of your repo.** Indexes are kept in your user cache directory, never in the project.
- **Chunks follow code structure.** Chunks split at top-level declarations, keep
  decorators and doc comments attached, and fall back to overlapping windows for long functions.
- **Respects `.gitignore`**, plus an optional `.codeseekignore`. It skips vendored code,
  build output, lockfiles, minified files and binaries.
- **Interactive shell.** Loads the model once, then answers each query in about 0.1 s.
- **Scriptable.** `--json` output and grep-style exit codes.
- **Optional MCP server** for coding agents. It's opt-in per project and never
  registers itself (see [below](#optional-mcp-server)).

## Install

Requires Python 3.10+ and one of two embedding backends.

**Option A: Ollama (recommended).** It's light to install, and searches take under a
second because the model stays loaded in the Ollama server.

```sh
# install Ollama from https://ollama.com, then:
ollama pull embeddinggemma-2
pip install https://github.com/NikhilBhatt-15/codeseek/releases/download/v0.1.0/codeseek-0.1.0-py3-none-any.whl
```

**Option B: PyTorch, in-process.** No other software is needed, but it's a much larger
install (about 2 GB) and each one-shot command pays about 20 s of startup.

```sh
pip install "codeseek[torch] @ https://github.com/NikhilBhatt-15/codeseek/releases/download/v0.1.0/codeseek-0.1.0-py3-none-any.whl"
```

The first run downloads the weights (about 1.5 GB) into the Hugging Face cache. On Linux
without a GPU, install the CPU build of PyTorch first to avoid pulling in CUDA:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

Add the `mcp` extra (`codeseek[mcp]`) for the optional MCP server.

The backend is chosen automatically. A new index uses Ollama if it's running and has
the model, and otherwise uses PyTorch. Each index remembers which backend built it and
is always searched with that one, because the two produce slightly different vectors
(Q8 vs fp32 weights). Override with `--backend ollama|torch`. Switching an existing
index requires `index --rebuild`.

From source:

```sh
git clone https://github.com/NikhilBhatt-15/codeseek && cd codeseek
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[torch,mcp,dev]"
```

## Usage

```sh
codeseek index                      # index the repo containing the current directory
codeseek index ~/src/project        # or a specific path
codeseek search "parse config file and apply defaults"
codeseek search "retry with backoff" -k 5 --glob "src/**/*.ts"
codeseek search "db connection pool" --json | jq '.[0].path'
codeseek search "..." --update      # refresh the index first
codeseek shell                      # interactive: fast repeated queries
codeseek status
codeseek clear
```

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | no results |
| 2 | usage error |
| 3 | repository not indexed, or index built with a different model, backend or dimension |
| 4 | embedding backend unavailable (Ollama not running or model not pulled; PyTorch not installed) |

### Configuration

Every setting can be set with an environment variable:

| Variable | Default | Notes |
|---|---|---|
| `CODESEEK_BACKEND` | `auto` | `ollama` or `torch`. Also available as `--backend`. |
| `CODESEEK_OLLAMA_URL` | `http://127.0.0.1:11434` | |
| `CODESEEK_OLLAMA_MODEL` | `embeddinggemma-2` | |
| `CODESEEK_OLLAMA_THREADS` | 3/4 of CPUs | Ollama's own default left cores idle: 2.0 s vs 1.35 s per chunk in testing. |
| `CODESEEK_DIM` | `256` | Embedding size: 768, 512, 256 or 128 (Matryoshka). Changing it requires `index --rebuild`. Also available as `--dim`. |
| `CODESEEK_DEVICE` | auto | PyTorch backend: `cuda`, `mps` or `cpu`. |
| `CODESEEK_BATCH_SIZE` | `16` | |
| `CODESEEK_MAX_FILE_BYTES` | `1000000` | Larger files are skipped. |
| `CODESEEK_INDEX_ROOT` | user cache dir | Where indexes are stored. |
| `CODESEEK_MODEL_ID` | `google/embeddinggemma-2` | |

### Performance

Measured on an Intel i5-12500H laptop, CPU only, indexing the same 4,700-line Rust project:

| | Ollama (Q8) | PyTorch (fp32) |
|---|---|---|
| First index | 1m54s (about 2,500 lines/min) | 2m43s (about 1,750 lines/min) |
| Re-index, nothing changed | under 1 s | under 1 s |
| One-shot `codeseek search` | **0.6 s** | about 20 s (mostly importing PyTorch and transformers) |
| Query in `codeseek shell` or MCP | about 0.1 s | about 0.13 s |
| Top-3 results for four test queries | the same three files for every query (two in a different order) | |

On CPU the first index of a large repository takes a while: about 20 minutes per 50k
lines with Ollama. After that, re-indexing is incremental. A GPU (used automatically by
both Ollama and PyTorch) is much faster.

Google's headline figure of about 190 MB RAM for this model is for a quantized LiteRT
build on a phone. The PyTorch backend holds the fp32 weights (about 1.7 GB resident).
bf16 and PyTorch's int8 dynamic quantization were both tried: neither was faster on a CPU
without native bf16 support.

## Optional MCP server

`codeseek-mcp` exposes one read-only tool, `search_code`, to MCP-capable coding agents.
**Nothing registers it automatically.** Coding agents already have good search tools, and
another tool in their context isn't always an improvement, so add it only to projects
where semantic search clearly helps (large or unfamiliar codebases).

To enable it for a single project, index the repo first, then add a project-scoped
config such as `.mcp.json` in that repo:

```json
{
  "mcpServers": {
    "codeseek": {
      "command": "codeseek-mcp",
      "args": ["--root", "."]
    }
  }
}
```

The server only reads the existing index and never builds or changes one. Run
`codeseek index` yourself to refresh it.

## How it works

1. **Discover.** Walk the repo, apply `.gitignore` and `.codeseekignore`, keep text source
   files, and skip binaries (NUL-byte sniff), lockfiles and minified bundles.
2. **Chunk.** Split each file at top-level declarations (`def`, `class`, `fn`, `func`,
   `impl`, `export`, …), pack neighbouring pieces into chunks of up to 50 lines or
   1,500 characters, and use overlapping windows for oversized blocks.
3. **Embed.** Each chunk is formatted with EmbeddingGemma's retrieval document prompt,
   `title: <path> | text: <code>`, embedded by the text encoder (via Ollama's
   `/api/embed` or in-process PyTorch), cut to 256 dimensions (Matryoshka) and
   L2-normalised. Queries use `task: code retrieval | query: …`.
4. **Store.** One SQLite database per repo, holding file hashes, chunk text and float32
   vectors.
5. **Search.** Exact cosine similarity, computed with a single NumPy matrix-vector product.
   This is exact (not approximate) and fast up to a few hundred thousand chunks.

## Development

```sh
pip install -e ".[torch,dev,mcp]"
ruff check . && ruff format --check .
mypy src tests
pytest                                    # fast suite using a deterministic fake embedder
CODESEEK_TEST_MODEL=1 pytest -m model     # end-to-end with the real model
```

## License

MIT for codeseek itself. EmbeddingGemma 2 is distributed by Google under Apache 2.0.
