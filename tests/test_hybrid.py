"""Unit tests: hybrid corpus — indexing, retrieval, add/remove, filters.

Models are stubbed (random, deterministic-seeded vectors) so the suite runs
fast and needs no network. ChromaDB persists to a per-test temp directory.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from app import config, hybrid

RNG = np.random.default_rng(7)


class StubEncoder:
    """Mimics SentenceTransformer.encode: list -> matrix, str/array -> vector."""

    def __init__(self, dim):
        self.dim = dim

    def encode(self, x, *a, **k):
        if isinstance(x, (list, tuple)):
            return RNG.normal(size=(len(x), self.dim)).astype(np.float32)
        return RNG.normal(size=self.dim).astype(np.float32)


@pytest.fixture
def corpus_env(tmp_path, monkeypatch):
    """Point DATA_INDEX/DATA_RAW at tmp dirs, stub the embedding models."""
    index = tmp_path / "index"
    raw = tmp_path / "raw"
    (index / "pages").mkdir(parents=True)
    raw.mkdir()
    monkeypatch.setattr(config, "DATA_INDEX", str(index))
    monkeypatch.setattr(config, "DATA_RAW", str(raw))

    # fake page images (two PNGs for doc A)
    for name in ("a_p001.png", "a_p002.png"):
        Image.new("RGB", (48, 48), "white").save(index / "pages" / name)

    pages = [
        {"doc": "a", "file": "a.pdf", "kind": "pdf", "page": 1, "section": "Intro",
         "text": "Regularization reduces overfitting in machine learning models. " * 20,
         "image": "pages/a_p001.png"},
        {"doc": "a", "file": "a.pdf", "kind": "pdf", "page": 2, "section": "L2",
         "text": "Ridge adds the sum of the squares of the coefficients. " * 20,
         "image": "pages/a_p002.png"},
        {"doc": "b", "file": "b.md", "kind": "md", "page": 1, "section": None,
         "text": "RAG retrieves relevant passages and adds them to the context.",
         "image": None},
    ]
    with open(index / "pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f)

    monkeypatch.setattr(hybrid, "_text_encoder", lambda: StubEncoder(384))
    monkeypatch.setattr(hybrid, "_clip_model", lambda: StubEncoder(512))
    return {"index": index, "raw": raw, "pages": pages}


def test_corpus_build_indexes(corpus_env):
    c = hybrid.Corpus(rebuild=True)
    assert len(c.pages) == 3
    assert c.text_coll.count() == len(c.chunks) > 0
    assert c.visual_coll.count() == 2  # only pages with images


def test_retrieve_returns_hits_and_doc_filter(corpus_env):
    c = hybrid.Corpus(rebuild=True)
    hits = c.retrieve("regularization", top_k=3)
    assert hits, "expected at least one hit"
    for h in hits:
        assert h["page"]["doc"] in ("a", "b")
        assert "chunk" in h and "score" in h
    docs_only = c.retrieve("regularization", top_k=5, doc="a")
    assert docs_only and all(h["page"]["doc"] == "a" for h in docs_only)


def test_ingest_and_index_adds_new_doc(corpus_env):
    c = hybrid.Corpus(rebuild=True)
    before = len(c.chunks)
    # production flow: the API saves the file into DATA_RAW before indexing it
    new_path = os.path.join(config.DATA_RAW, "c.md")
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("Gradient clipping prevents exploding gradients. " * 10)
    res = c.ingest_and_index([new_path])
    assert res["pages"] == 4
    assert os.path.exists(new_path)
    assert len(c.chunks) > before
    assert c.text_coll.count() == len(c.chunks)


def test_remove_document_purges_everything(corpus_env):
    c = hybrid.Corpus(rebuild=True)
    res = c.remove_document("a")
    assert res is not None
    assert res["pages"] == 1
    assert all(p["doc"] != "a" for p in c.pages)
    assert not any(ch["doc"] == "a" for ch in c.chunks)
    assert c.text_coll.count() == len(c.chunks)
    assert c.visual_coll.count() == 0
    for p in corpus_env["pages"]:
        img = p.get("image")
        if img and p["doc"] == "a":
            assert not os.path.exists(os.path.join(config.DATA_INDEX, img))
    assert not os.path.exists(os.path.join(config.DATA_RAW, "a.pdf"))
    # removing an unknown doc -> None
    assert c.remove_document("nope") is None


def test_duplicate_rejected_on_ingest(corpus_env, tmp_path):
    c = hybrid.Corpus(rebuild=True)
    dup = tmp_path / "a.pdf"
    dup.write_text("junk-not-real-pdf", encoding="utf-8")
    with pytest.raises(ValueError, match="Already indexed"):
        c.ingest_and_index([str(dup)])
