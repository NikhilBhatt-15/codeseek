"""End-to-end check against the real model. Skipped unless CODESEEK_TEST_MODEL=1."""

import os
from pathlib import Path

import numpy as np
import pytest

from codeseek.config import Settings
from codeseek.embedder import GemmaEmbedder
from codeseek.service import CodeSeek

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("CODESEEK_TEST_MODEL") != "1", reason="set CODESEEK_TEST_MODEL=1"
    ),
]


def test_real_model_semantic_search(repo: Path, settings: Settings) -> None:
    embedder = GemmaEmbedder(settings)
    vec = embedder.embed_query("hello")
    assert vec.shape == (settings.dim,)
    assert np.isfinite(vec).all()
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3

    with CodeSeek(repo, settings, embedder) as cs:
        cs.index()
        # No shared keywords with the code: this only passes with real semantics.
        assert cs.search("keep users signed in when credentials expire", k=1)[0].path == (
            "src/auth.py"
        )
        assert cs.search("connect to SQL server", k=1)[0].path == "src/db.go"
