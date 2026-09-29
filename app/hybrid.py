"""Hybrid RAG retrieval index.

Three complementary indexes over the course pages, kept separate per the spec:
  1. KEYWORD  — BM25 (bm25s) over text chunks
  2. TEXT     — semantic text embeddings (all-MiniLM) in a chromadb collection
  3. VISUAL   — CLIP image embeddings of every page/slide image in a chromadb
                collection, queried cross-modally by the text question

Pages are split into overlapping chunks (langchain-text-splitters), each
preserving source details (doc, file, page, section). Retrieval runs all three
indexes, combines candidates by Reciprocal Rank Fusion (rerank), and returns the
best text chunks together with their original page/slide images.

Indexes are cached in data/index/ (gitignored) so server startup is fast.
"""
import json
import os

import numpy as np

from . import config

# ---------------------------------------------------------------------------
# lazy model singletons
# ---------------------------------------------------------------------------
_text_enc = None
_clip = None
_corpus = None


def _text_encoder():
    global _text_enc
    if _text_enc is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading text embedding model: {config.EMBED_MODEL}")
        _text_enc = SentenceTransformer(config.EMBED_MODEL)
    return _text_enc


def _clip_model():
    global _clip
    if _clip is None:
        from sentence_transformers import SentenceTransformer
        print("Loading CLIP model: clip-ViT-B-32")
        _clip = SentenceTransformer("clip-ViT-B-32")
    return _clip


def load_pages():
    path = os.path.join(config.DATA_INDEX, "pages.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chunk_pages(pages):
    """Split page text into overlapping chunks, preserving source details.

    Chunk ids are DETERMINISTIC (doc__page__segment) so re-chunking and
    incremental uploads keep stable identifiers across index updates.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    page_index_by_key = {(p["doc"], p["page"], p["file"]): i for i, p in enumerate(pages)}
    chunks = []
    for p in pages:
        text = (p.get("text") or "").strip()
        pieces = splitter.split_text(text) if text else [""]
        for seg, piece in enumerate(pieces):
            chunks.append({
                "cid": f"{p['doc']}__p{p['page']}__{seg}",
                "doc": p["doc"],
                "file": p["file"],
                "kind": p["kind"],
                "page": p["page"],
                "section": p.get("section"),
                "text": piece.strip(),
                "image": p.get("image"),
                "seg": seg,
                "page_index": page_index_by_key[(p["doc"], p["page"], p["file"])],
            })
    return chunks


# ---------------------------------------------------------------------------
# visual (image) helpers
# ---------------------------------------------------------------------------
def _page_image_path(page):
    if not page.get("image"):
        return None
    return os.path.join(config.DATA_INDEX, page["image"])


def _visual_id(page):
    return f"{page['doc']}__p{page['page']}"


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
class Corpus:
    def __init__(self, rebuild=False):
        import threading
        self._lock = threading.Lock()
        self.pages = load_pages()
        self.chunks = self._load_chunks(rebuild)
        self.text_enc = _text_encoder()
        self._init_chroma()
        self._build_keyword()          # small corpus -> build in memory each start
        if rebuild:
            self._build_text_vectors()
            self._build_visual_vectors()

    # ---- persistence ---------------------------------------------------
    def _chunks_path(self):
        return os.path.join(config.DATA_INDEX, "chunks.json")

    def _load_chunks(self, rebuild):
        p = self._chunks_path()
        if rebuild or not os.path.exists(p):
            chunks = _chunk_pages(self.pages)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            return chunks
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _init_chroma(self):
        import chromadb
        self.client = chromadb.PersistentClient(
            path=os.path.join(config.DATA_INDEX, "chroma"))
        cosine = {"hnsw:space": "cosine"}
        self.text_coll = self.client.get_or_create_collection("text", metadata=cosine)
        self.visual_coll = self.client.get_or_create_collection("visual", metadata=cosine)

    def _all_texts(self):
        return [c["text"] for c in self.chunks]

    def _flu_clear(self, coll):
        ids = coll.get()["ids"]
        if ids:
            coll.delete(ids=ids)

    # ---- index construction --------------------------------------------
    def _build_keyword(self):
        import bm25s
        tok = bm25s.tokenize(self._all_texts(), stopwords="en")
        self.retriever = bm25s.BM25()
        self.retriever.index(tok)

    def _build_text_vectors(self):
        vecs = self.text_enc.encode(self._all_texts(), batch_size=32)
        ids = [c["cid"] for c in self.chunks]
        metas = [{"doc": c["doc"], "page": int(c["page"])}
                 for c in self.chunks]
        self._flu_clear(self.text_coll)
        self.text_coll.add(ids=ids, embeddings=vecs.tolist(), metadatas=metas,
                           documents=self._all_texts())

    def _build_visual_vectors(self):
        clip = _clip_model()
        rows = []
        for p in self.pages:
            img_path = _page_image_path(p)
            if not img_path:
                continue
            emb = clip.encode(_image_to_rgb_array(img_path))
            rows.append({"id": _visual_id(p), "emb": emb,
                         "meta": {"doc": p["doc"], "page": int(p["page"])}})
        self._flu_clear(self.visual_coll)
        self.visual_coll.add(ids=[r["id"] for r in rows],
                             embeddings=[r["emb"].tolist() for r in rows],
                             metadatas=[r["meta"] for r in rows])

    # ---- incremental upload indexing -----------------------------------
    def ingest_and_index(self, file_paths):
        """Ingest new course files and incrementally extend all three indexes.

        Existing pages/chunks/images keep their ids (deterministic), so only the
        NEW content is embedded and upserted. Returns {"added", "pages", "chunks"}.
        """
        from . import ingest
        with self._lock:
            new_pages = []
            for fp in file_paths:
                ps = ingest.ingest_file(fp)
                if ps:
                    new_pages.extend(ps)
            if not new_pages:
                raise ValueError("No supported content found in the uploaded file(s).")

            # reject documents already in the index (same slug = same file)
            existing_docs = {p["doc"] for p in self.pages}
            dups = [p["file"] for p in new_pages if p["doc"] in existing_docs]
            if dups:
                raise ValueError(f"Already indexed (rename or delete first): {dups[0]}")

            merged = self.pages + new_pages
            with open(os.path.join(config.DATA_INDEX, "pages.json"), "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

            chunks = _chunk_pages(merged)
            with open(self._chunks_path(), "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.pages, self.chunks = merged, chunks

            self._build_keyword()
            self._add_new_text_vectors()
            self._add_new_visual_vectors(new_pages)
            return {"added": [p["file"] for p in new_pages],
                    "pages": len(self.pages), "chunks": len(self.chunks)}

    def _add_new_text_vectors(self):
        """Embed + upsert only chunks not already in the text collection."""
        existing = set(self.text_coll.get()["ids"])
        new = [c for c in self.chunks if c["cid"] not in existing]
        if not new:
            return
        vecs = self.text_enc.encode([c["text"] for c in new], batch_size=32)
        self.text_coll.upsert(
            ids=[c["cid"] for c in new],
            embeddings=vecs.tolist(),
            metadatas=[{"doc": c["doc"], "page": int(c["page"])} for c in new],
            documents=[c["text"] for c in new],
        )

    def _add_new_visual_vectors(self, new_pages):
        """Embed + upsert only page images not already in the visual collection."""
        existing = set(self.visual_coll.get()["ids"])
        clip = _clip_model()
        rows = []
        for p in new_pages:
            if _visual_id(p) in existing:
                continue
            img_path = _page_image_path(p)
            if not img_path or not os.path.exists(img_path):
                continue
            emb = clip.encode(_image_to_rgb_array(img_path))
            rows.append({"id": _visual_id(p), "emb": emb,
                         "meta": {"doc": p["doc"], "page": int(p["page"])}})
        if rows:
            self.visual_coll.upsert(ids=[r["id"] for r in rows],
                                    embeddings=[r["emb"].tolist() for r in rows],
                                    metadatas=[r["meta"] for r in rows])

    # ---- filtering helpers ---------------------------------------------
    def _candidate_chunk_ids(self, doc, topic):
        ids = list(range(len(self.chunks)))
        if doc:
            ids = [i for i in ids if self.chunks[i]["doc"] == doc]
        if topic:
            t = topic.lower()
            ids = [i for i in ids
                   if t in self.chunks[i]["text"].lower()
                   or t in (self.chunks[i]["section"] or "").lower()]
        return ids

    def _candidate_page_set(self, doc, topic):
        return {self.chunks[i]["page_index"] for i in self._candidate_chunk_ids(doc, topic)}

    def _page_index_by_visual_id(self, vid):
        doc, page = vid.rsplit("__p", 1)
        for i, p in enumerate(self.pages):
            if p["doc"] == doc and p["page"] == int(page):
                return i
        return None

    def _visual_page_hits(self, query, doc, top_n=20):
        """Cross-modal text->image query over the visual index."""
        if not self.visual_coll.count():
            return []
        clip = _clip_model()
        qv = clip.encode(query).tolist()
        where = {"doc": doc} if doc else None
        res = self.visual_coll.query(query_embeddings=[qv], n_results=min(top_n, self.visual_coll.count()),
                                     where=where)
        return [i for i in (self._page_index_by_visual_id(v) for v in res["ids"][0]) if i is not None]

    # ---- retrieval -----------------------------------------------------
    def retrieve(self, query, top_k=None, doc=None, topic=None):
        top_k = top_k or config.RETRIEVE_TOP_K
        if not self.chunks:
            return []
        cand_i = self._candidate_chunk_ids(doc, topic)
        cand_set = set(cand_i)
        cand_cids = {self.chunks[i]["cid"] for i in cand_i}
        cand_pages = self._candidate_page_set(doc, topic)
        if not cand_set:
            return []

        import bm25s
        ranked = []  # list of ranked chunk-id lists

        # 1) keyword (BM25) over full index, then filtered to candidates
        res = self.retriever.retrieve(bm25s.tokenize([query], stopwords="en"),
                                      k=min(len(self.chunks), 100))
        # `documents` == corpus indices when no corpus is attached
        kw_ids = [self.chunks[int(i)]["cid"] for i in res.documents[0] if int(i) in cand_set]
        if kw_ids:
            ranked.append(kw_ids)

        # 2) text embeddings (chromadb), filtered to candidates
        qv = self.text_enc.encode(query).tolist()
        where = {"doc": doc} if doc else None
        try:
            res = self.text_coll.query(query_embeddings=[qv],
                                       n_results=min(60, max(len(cand_set), 1)),
                                       where=where)
            txt_ids = [cid for cid in res["ids"][0] if cid in cand_cids]
            if txt_ids:
                ranked.append(txt_ids)
        except Exception:
            pass

        # 3) visual embeddings (CLIP) — cross-modal, then mapped to candidate pages
        try:
            v_pages = self._visual_page_hits(query, doc)
            v_pages = [p for p in v_pages if p in cand_pages]
            v_ids = [self.chunks[i]["cid"] for i in range(len(self.chunks))
                     if self.chunks[i]["page_index"] in v_pages]
            if v_ids:
                ranked.append(v_ids)
        except Exception:
            pass

        # Reciprocal Rank Fusion (rerank across the three lists)
        rrf, k = {}, 60
        for lst in ranked:
            for rank, item in enumerate(lst):
                rrf[item] = rrf.get(item, 0) + 1.0 / (k + rank + 1)
        if not rrf:
            return []
        ordered = sorted(rrf, key=rrf.get, reverse=True)

        # collapse to pages (best chunk per page), ordered by RRF score
        cid_to_i = {c["cid"]: i for i, c in enumerate(self.chunks)}
        best_chunk = {}
        for cid in ordered:
            i = cid_to_i.get(cid)
            if i is None:
                continue
            pi = self.chunks[i]["page_index"]
            if pi not in best_chunk:
                best_chunk[pi] = (cid, rrf[cid])
        ordered_pages = sorted(best_chunk.items(), key=lambda kv: -kv[1][1])[:top_k]

        hits = []
        for pi, (cid, score) in ordered_pages:
            i = cid_to_i.get(cid)
            if i is None:
                continue
            ch = self.chunks[i]
            hits.append({
                "page": {**self.pages[pi], "image_source": None},
                "chunk": ch["text"],
                "score": float(score),
            })
        return hits


def get_corpus(rebuild=False):
    global _corpus
    if _corpus is None or rebuild:
        _corpus = Corpus(rebuild=rebuild)
    return _corpus


def rebuild_index():
    get_corpus(rebuild=True)


def _image_to_rgb_array(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return np.asarray(im)
