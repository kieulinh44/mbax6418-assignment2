"""Hybrid RAG retrieval index.

Three complementary indexes over the course pages, kept separate per the spec:
  1. KEYWORD  — BM25 (bm25s) over text chunks
  2. TEXT     — semantic text embeddings (all-MiniLM) in a chromadb collection
  3. VISUAL   — CLIP image embeddings of every page/slide image in a chromadb
                collection, queried cross-modally by the text question

Pages are split into overlapping chunks (langchain-text-splitters), each
preserving source details (doc, file, page, section). Retrieval runs all three
indexes, combines candidates by weighted score fusion (min-max normalized BM25
+ text-cosine + visual-cosine), and returns the best text chunks together with
their original page/slide images.

Indexes are cached in data/index/ (gitignored) so server startup is fast.
"""
import json
import logging
import os
import re

import numpy as np

from . import config

log = logging.getLogger("course-assistant")

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


def _atomic_write_json(path, obj):
    """Write JSON atomically (tmp + rename) so concurrent readers never see
    a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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
_IMAGE_QUERY_WORDS = ("diagram", "chart", "graph", "meme", "slide image", "picture",
                      "visual", "image", "illustration", "screenshot", "figure",
                      "what is on the slide", "describe the slide", "show me the")
RETRIEVAL_MODES = ("hybrid", "text_keyword_only")


def normalize_retrieval_mode(mode):
    value = (mode or "hybrid").strip().lower()
    if value not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of: {', '.join(RETRIEVAL_MODES)}")
    return value


def lexical_match_score(query, text):
    """Give exact query-term matches a small deterministic ranking boost."""
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_terms:
        return 0.0
    matched = query_terms & text_terms
    score = float(len(matched))
    if "grading" in query_terms and ({"grading", "percentage", "percent"} & text_terms):
        score += 2.0
    if {"breakdown", "grade", "grades"} & query_terms and ({"quiz", "quizzes", "exam", "project", "attendance"} & text_terms):
        score += 2.0
    return score


def _is_image_query(query):
    q = query.lower()
    return any(w in q for w in _IMAGE_QUERY_WORDS)


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
            _atomic_write_json(p, chunks)
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
            # reject documents already in the index BEFORE parsing anything
            existing_docs = {p["doc"] for p in self.pages}
            for fp in file_paths:
                if ingest._slug(os.path.basename(fp)) in existing_docs:
                    raise ValueError(f"Already indexed (rename or delete first): {os.path.basename(fp)}")

            new_pages = []
            for fp in file_paths:
                ps = ingest.ingest_file(fp)
                if ps:
                    new_pages.extend(ps)
            if not new_pages:
                raise ValueError("No supported content found in the uploaded file(s).")

            merged = self.pages + new_pages
            _atomic_write_json(os.path.join(config.DATA_INDEX, "pages.json"), merged)

            chunks = _chunk_pages(merged)
            _atomic_write_json(self._chunks_path(), chunks)
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

    def _visual_page_hits(self, query, doc, top_n=20, where=None):
        """Cross-modal text->image query over the visual index.
        Returns [(page_index, similarity), ...] ordered by similarity."""
        if not self.visual_coll.count():
            return []
        clip = _clip_model()
        qv = clip.encode(query).tolist()
        if where is None:
            where = {"doc": doc} if doc else None
        res = self.visual_coll.query(query_embeddings=[qv], n_results=min(top_n, self.visual_coll.count()),
                                     where=where)
        out = []
        ids = res.get("ids") or []
        distances = res.get("distances") or []
        if not ids or not distances or not ids[0] or not distances[0]:
            return out
        for vid, dist in zip(ids[0], distances[0]):
            pi = self._page_index_by_visual_id(vid)
            if pi is not None:
                out.append((pi, 1.0 - float(dist)))
        return out

    # ---- removal ---------------------------------------------------------
    def remove_document(self, doc):
        """Remove a document and ALL of its searchable content: pages, chunks,
        text/visual embeddings, keyword hits, rendered images, and the raw file.
        Returns {"removed", "pages", "chunks"} or None if the doc is unknown."""
        with self._lock:
            removed_pages = [p for p in self.pages if p["doc"] == doc]
            if not removed_pages:
                return None
            remaining = [p for p in self.pages if p["doc"] != doc]

            # chunks to drop: deterministic prefix doc__p
            keep_chunks = [c for c in self.chunks if c["doc"] != doc]
            drop_cids = [c["cid"] for c in self.chunks if c["doc"] == doc]
            drop_visual = [_visual_id(p) for p in removed_pages]

            _atomic_write_json(os.path.join(config.DATA_INDEX, "pages.json"), remaining)
            _atomic_write_json(self._chunks_path(), keep_chunks)

            # vectors
            to_del_txt = [c for c in drop_cids if c in self.text_coll.get()["ids"]]
            if to_del_txt:
                self.text_coll.delete(ids=to_del_txt)
            to_del_vis = [v for v in drop_visual if v in self.visual_coll.get()["ids"]]
            if to_del_vis:
                self.visual_coll.delete(ids=to_del_vis)

            # keyword (rebuilt in memory over remaining chunks)
            self.chunks, self.pages = keep_chunks, remaining
            self._build_keyword()

            # rendered page/slide images
            for p in removed_pages:
                if p.get("image"):
                    img = os.path.join(config.DATA_INDEX, p["image"])
                    if os.path.exists(img):
                        try:
                            os.remove(img)
                        except OSError:
                            pass

            # raw source file
            for p in removed_pages:
                raw = os.path.join(config.DATA_RAW, p["file"])
                if os.path.exists(raw) and os.path.isfile(raw):
                    try:
                        os.remove(raw)
                    except OSError:
                        pass

            return {"removed": doc, "pages": len(remaining), "chunks": len(keep_chunks)}

    # ---- retrieval -----------------------------------------------------
    def retrieve(self, query, top_k=None, doc=None, topic=None, retrieval_mode="hybrid"):
        """Thread-safe retrieval: serializes against upload index mutation."""
        with self._lock:
            return self._retrieve_unlocked(query, top_k=top_k, doc=doc, topic=topic,
                                           retrieval_mode=retrieval_mode)

    def _retrieve_unlocked(self, query, top_k=None, doc=None, topic=None,
                           retrieval_mode="hybrid"):
        retrieval_mode = normalize_retrieval_mode(retrieval_mode)
        top_k = top_k or config.RETRIEVE_TOP_K
        if not self.chunks:
            return []
        cand_i = self._candidate_chunk_ids(doc, topic)
        requested_page = None
        # Only an unambiguous singular reference narrows the search. Multiple
        # references, numeric lists/ranges and decimals keep normal hybrid RAG.
        references = list(re.finditer(
            r"\b(?:slide|page)\s+(?:number\s+|#\s*)?(\d+)\b",
            query, re.IGNORECASE))
        if len(references) == 1:
            requested = references[0]
            continuation = query[requested.end():]
            ambiguous = re.match(
                r"\s*(?:[-–—,./&]|\b(?:to|through|and|or)\b)\s*#?\s*\d",
                continuation, re.IGNORECASE)
            if not ambiguous:
                requested_page = int(requested.group(1))
                cand_i = [i for i in cand_i if self.chunks[i]["page"] == requested_page]
        cand_set = set(cand_i)
        cand_cids = {self.chunks[i]["cid"] for i in cand_i}
        cand_pages = {self.chunks[i]["page_index"] for i in cand_i}
        cid_to_i = {c["cid"]: i for i, c in enumerate(self.chunks)}
        if not cand_set:
            return []

        # Apply explicit page constraints inside Chroma, before bounded top-k.
        where = {"doc": doc} if doc else None
        if requested_page is not None:
            page_filter = {"page": requested_page}
            where = {"$and": [where, page_filter]} if where else page_filter

        import bm25s
        fused = {}  # cid -> weighted fused score

        def add_stage(scores, weight):
            """Min-max normalize one stage's raw scores, then add weighted."""
            if not scores:
                return
            vals = list(scores.values())
            lo, hi = min(vals), max(vals)
            span = hi - lo
            for cid, s in scores.items():
                n = (s - lo) / span if span > 1e-9 else 1.0
                fused[cid] = fused.get(cid, 0.0) + weight * n

        # 1) keyword (BM25) — raw scores preserve dominance
        res = self.retriever.retrieve(bm25s.tokenize([query], stopwords="en"),
                                      k=(len(self.chunks) if requested_page is not None
                                         else min(len(self.chunks), 100)))
        kw = {}
        documents = getattr(res, "documents", None)
        scores = getattr(res, "scores", None)
        if documents is None:
            documents = []
        if scores is None:
            scores = []
        if len(documents) > 0 and len(scores) > 0 and len(documents[0]) > 0 and len(scores[0]) > 0:
            for i, s in zip(documents[0], scores[0]):
                i = int(i)
                if i in cand_set:
                    kw[self.chunks[i]["cid"]] = float(s)
        for i in cand_i:
            boost = lexical_match_score(query, self.chunks[i]["text"])
            if boost:
                cid = self.chunks[i]["cid"]
                kw[cid] = kw.get(cid, 0.0) + boost
        add_stage(kw, 1.0)

        # 2) text embeddings (chromadb), filtered to candidates
        qv = self.text_enc.encode(query).tolist()
        txt = {}
        try:
            res = self.text_coll.query(query_embeddings=[qv],
                                       n_results=min(60, max(len(cand_set), 1)),
                                       where=where)
            ids = res.get("ids") or []
            distances = res.get("distances") or []
            if ids and distances and ids[0] and distances[0]:
                txt = {cid: 1.0 - float(d) for cid, d in zip(ids[0], distances[0])
                       if cid in cand_cids}
            add_stage(txt, 1.0)
        except Exception:
            log.exception("chroma text-embedding query failed")

        # 3) visual embeddings (CLIP) — cross-modal. For text questions, visual
        #    only BOOSTS pages already surfaced by keyword/text (avoids irrelevant
        #    diagram slides dominating); for explicit image questions visual leads.
        try:
            image_query = _is_image_query(query) if retrieval_mode == "hybrid" else False
            vis_weight = (1.0 if image_query else 0.35) if retrieval_mode == "hybrid" else 0.0
            v_pages = self._visual_page_hits(query, doc, where=where) if retrieval_mode == "hybrid" else []
            if not image_query:
                surfaced = set()
                for cid in kw:
                    i = cid_to_i.get(cid)
                    if i is not None:
                        surfaced.add(self.chunks[i]["page_index"])
                for cid in txt:
                    i = cid_to_i.get(cid)
                    if i is not None:
                        surfaced.add(self.chunks[i]["page_index"])
                v_pages = [(pi, s) for pi, s in v_pages if pi in surfaced]
            v_pages = [(pi, s) for pi, s in v_pages if pi in cand_pages]
            page_chunks = {}
            for i in cand_i:
                ch = self.chunks[i]
                page_chunks.setdefault(ch["page_index"], []).append(ch["cid"])
            vis = {}
            for pi, sim in v_pages:
                for cid in page_chunks.get(pi, []):
                    if image_query or cid in fused:
                        vis[cid] = max(vis.get(cid, 0.0), sim)
            # image queries: ensure a visual-only page still contributes its chunks
            if not vis:
                for pi, sim in v_pages:
                    for cid in page_chunks.get(pi, []):
                        vis[cid] = max(vis.get(cid, 0.0), sim)
            add_stage(vis, vis_weight)
        except Exception:
            log.exception("chroma visual-embedding query failed")

        if not fused:
            return []
        ordered = sorted(fused, key=fused.get, reverse=True)

        # collapse to pages (best chunk per page), ordered by fused score
        best_chunk = {}
        for cid in ordered:
            i = cid_to_i.get(cid)
            if i is None:
                continue
            pi = self.chunks[i]["page_index"]
            if pi not in best_chunk:
                best_chunk[pi] = (cid, fused[cid])
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
