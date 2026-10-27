# Changelog

## 0.1.0

First release.

- `index`, `search`, `shell`, `status`, `clear` commands.
- EmbeddingGemma 2 text encoder (270M), 256-dimension Matryoshka vectors by default.
- Two backends: Ollama (recommended, sub-second searches) and in-process PyTorch
  (`codeseek[torch]` extra). Chosen automatically and recorded per index.
- Incremental, hash-checked indexing stored in the user cache directory.
- Chunks that follow code structure, with overlapping windows for long blocks.
- `.gitignore` / `.codeseekignore` support.
- Optional, opt-in MCP server (`codeseek-mcp`) with a read-only `search_code` tool.
