"""Local embeddings + vector retrieval over page documents.

Pages are embedded into fixed-size vectors and matched by cosine similarity.
Small enough that an in-memory numpy index is fine (no external vector DB).
"""
import json
import os

import numpy as np

from . import config

# Lazy model load (sentence-transformers pulls torch the first time)
_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {config.EMBED_MODEL}")
        _encoder = SentenceTransformer(config.EMBED_MODEL)
    return _encoder


def load_pages():
    path = os.path.join(config.DATA_INDEX, "pages.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(pages):
    """Compute + cache embeddings for page texts. Returns matrix (n, dim)."""
    vec_path = os.path.join(config.DATA_INDEX, "vectors.npy")
    texts = [p.get("text", "") for p in pages]
    if os.path.exists(vec_path):
        vecs = np.load(vec_path)
        if len(vecs) == len(pages):
            return vecs
    enc = encoder()
    # encode in batches to avoid a giant batch
    vecs = np.vstack([enc.encode(batch) for batch in _batches(texts, 32)])
    np.save(vec_path, vecs)
    return vecs


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def retrieve(query, top_k=None, doc=None, topic=None):
    """Return top-k pages for a query (semantic similarity)."""
    top_k = top_k or config.RETRIEVE_TOP_K
    pages = load_pages()
    if not pages:
        return []
    idx = np.arange(len(pages))
    if doc:
        idx = idx[np.array([p["doc"] == doc for p in pages])]
    # topic filter: narrow to pages whose section/text mention the topic
    if topic:
        topic = topic.lower()
        keep = [p["doc"] == doc or topic in (p.get("section") or "").lower()
                or topic in p.get("text", "").lower()
                for p in pages]
        idx = idx[np.array(keep)]
    if len(idx) == 0:
        return []
    vecs = build_index(pages)
    enc = encoder()
    qv = enc.encode(query)
    qv = qv / np.linalg.norm(qv)
    sub_pages = [pages[i] for i in idx]
    sub_vecs = vecs[idx]
    norms = np.linalg.norm(sub_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sims = (sub_vecs @ qv) / norms.ravel()
    order = np.argsort(-sims)[:top_k]
    return [{"page": {**sub_pages[i], "image_source": None},
             "score": float(sims[i])}
            for i in order]
