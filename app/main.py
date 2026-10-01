"""FastAPI app: ingest-facing file list + grounded Q&A endpoint."""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config, llm, quiz, hybrid

app = FastAPI(title="Course Assistant")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
                   allow_methods=["*"], allow_headers=["*"])

# Serve page images (rendered copies) and the front-end
os.makedirs(os.path.join(config.DATA_INDEX, "pages"), exist_ok=True)
app.mount("/pages", StaticFiles(directory=os.path.join(config.DATA_INDEX, "pages")), name="pages")

STATIC_DIR = os.path.join(config.BASE_DIR, "static")


@app.get("/", response_class=HTMLResponse)
def home():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return "Course assistant UI not built yet."


@app.get("/api/files")
def list_files():
    pages = hybrid.load_pages()
    by_doc = {}
    for p in pages:
        d = by_doc.setdefault(p["doc"], {"file": p["file"], "kind": p["kind"],
                                         "pages": 0, "segments": 0})
        d["pages"] = max(d["pages"], p["page"])
        d["segments"] += 1
    return {"documents": [{"doc": k, **v} for k, v in by_doc.items()]}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/materials")
def list_materials():
    """List course material files available for download (from data/raw)."""
    from urllib.parse import quote
    items = []
    if os.path.isdir(config.DATA_RAW):
        for fname in sorted(os.listdir(config.DATA_RAW)):
            fpath = os.path.join(config.DATA_RAW, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                items.append({
                    "file": fname,
                    "kind": os.path.splitext(fname)[1].lstrip(".").lower() or "?",
                    "size": os.path.getsize(fpath),
                    "url": f"/api/download/{quote(fname)}",
                })
    return {"materials": items}


@app.get("/api/download/{fname}")
def download_material(fname: str):
    """Serve a course material file from data/raw (path-traversal safe)."""
    safe = os.path.basename(fname)
    fpath = os.path.join(config.DATA_RAW, safe)
    if not os.path.isfile(fpath):
        raise HTTPException(404, "File not found")
    return FileResponse(fpath, media_type="application/octet-stream", filename=safe)


@app.post("/api/upload")
async def upload_materials(files: list[UploadFile] = File(...)):
    """Upload course materials from the dashboard; ingests + indexes them."""
    from . import ingest as ingest_mod
    # Reject duplicates BEFORE touching disk so an existing source is never
    # overwritten by a rejected upload.
    existing_docs = {p["doc"] for p in hybrid.load_pages()}
    saved = []
    MAX_BYTES = 250 * 1024 * 1024  # per-file cap
    os.makedirs(config.DATA_RAW, exist_ok=True)
    try:
        for up in files:
            fname = os.path.basename(up.filename or "").replace("\\", "/").split("/")[-1]
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ingest_mod.SUPPORTED_EXTS:
                raise HTTPException(400, f"Unsupported format '{ext or '(none)'}'. "
                                         f"Accepted: {', '.join(sorted(ingest_mod.SUPPORTED_EXTS))}")
            if ingest_mod._slug(fname) in existing_docs:
                raise HTTPException(409, f"'{fname}' is already indexed "
                                         "(rename the file to add it as a new document).")
            dest = os.path.join(config.DATA_RAW, fname)
            data = await up.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, f"'{fname}' exceeds the {MAX_BYTES // (1024 * 1024)} MB upload limit.")
            with open(dest, "wb") as out:
                out.write(data)
            saved.append(dest)
        result = hybrid.get_corpus().ingest_and_index(saved)
        return {"ok": True, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        # never orphan a partially-uploaded file
        for p in saved:
            try:
                os.remove(p)
            except OSError:
                pass
        raise HTTPException(500, f"Indexing failed: {e}")


@app.delete("/api/documents/{doc}")
def remove_document(doc: str):
    """Remove a document and all of its searchable content (pages, chunks,
    embeddings, images, and the source file)."""
    result = hybrid.get_corpus().remove_document(doc)
    if result is None:
        raise HTTPException(404, f"Document '{doc}' not found in the index.")
    return {"ok": True, "result": result}


def _cite(page, excerpt=None):
    return {
        "doc": page["doc"],
        "file": page["file"],
        "page": page["page"],
        "section": page.get("section"),
        "excerpt": (excerpt or page.get("text") or "")[:900],
        "image": f"/pages/{os.path.basename(page['image'])}" if page.get("image") else None,
    }


def _validate(answer, hits):
    """Check that citations actually reference the retrieved evidence."""
    evidence = {(h["page"]["file"], h["page"]["page"]) for h in hits}
    cited = set()
    for file_, page in evidence:
        base = os.path.splitext(file_)[0]
        if base in answer or file_ in answer:
            cited.add((file_, page))
    sup = sorted(cited)
    unc = sorted(evidence - cited)
    return {
        "checked": len(evidence),
        "sources_used": [{"file": f, "page": p} for f, p in sup],
        "sources_not_cited": [{"file": f, "page": p} for f, p in unc],
        "all_sources_supported": len(unc) == 0,
    }


@app.post("/api/ask")
def ask(payload: dict):
    question = (payload.get("question") or "").strip()
    doc = payload.get("doc") or None
    topic = payload.get("topic") or None
    if not question:
        raise HTTPException(400, "question is required")

    hits = hybrid.get_corpus().retrieve(question, doc=doc, topic=topic)
    if not hits:
        return {
            "answer": "I could not find any matching material in the course files. "
                      "If this topic is covered, please ensure the documents are "
                      "ingested (run ingest) and check the file/filter selection.",
            "sources": [],
            "vision_notes": "",
            "validation": {"checked": 0, "all_sources_supported": True},
            "retrieval": "hybrid(keyword+text+visual)",
        }

    # Best evidence: top chunks as context, original page images as visual evidence
    context_blocks = []
    for h in hits:
        p = h["page"]
        context_blocks.append(
            f"[Document: {p['file']} | page/slide {p['page']}"
            + (f" | section: {p['section']}" if p.get("section") else "")
            + "]\n" + (h["chunk"] or "")
        )
    context = "\n\n".join(context_blocks)

    # Vision pass over the images of the top retrieved pages (diagrams/charts).
    # Sources below always carry the original slide image + doc name + slide number.
    vision_notes = ""
    img_paths = []
    for h in hits[:3]:
        if h["page"].get("image"):
            ip = os.path.join(config.DATA_INDEX, h["page"]["image"])
            if os.path.exists(ip):
                img_paths.append(ip)
    if img_paths:
        try:
            vision_notes = llm.vision(
                "Describe exactly what these page/slide images show — especially any "
                "diagram, chart, graph, or formula. Be concise and factual. If an image "
                "is mostly text or blank, say so.",
                img_paths,
            )
        except Exception:
            vision_notes = ""

    sys = (
        "You are a course assistant. Answer ONLY from the provided course material "
        "context. Every factual point must be grounded in the cited pages. Cite each "
        "source as [file p.N] where you use it. If the material does not contain the "
        "answer, say clearly that the information is not in the materials and DO NOT "
        "invent facts or citations. A vision description of the top-relevant images is "
        "included where available."
    )
    user = (f"QUESTION: {question}\n\n"
            + (f"VISION NOTES (top-relevant diagrams/images):\n{vision_notes}\n\n" if vision_notes else "")
            + "COURSE MATERIAL:\n" + context
            + "\n\nAnswer concisely, cite sources inline as [file p.N], and end with a "
              "'Sources:' section listing the file, page/slide, and a short reason each is relevant.")

    try:
        answer = llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {e}")

    return {
        "answer": answer,
        "sources": [_cite(h["page"], excerpt=h["chunk"]) for h in hits],
        "vision_notes": vision_notes,
        "validation": _validate(answer, hits),
        "retrieval": "hybrid(keyword+text+visual)",
    }


@app.post("/api/quiz")
def make_quiz(payload: dict):
    """Generate a practice quiz; the answer key stays server-side (not returned)."""
    doc = payload.get("doc") or None
    topic = payload.get("topic") or None
    theme = payload.get("theme") or None
    try:
        n = int(payload.get("n") or 4)
    except (TypeError, ValueError):
        raise HTTPException(400, "n must be a number (2–6).")
    try:
        q = quiz.generate_quiz(question_theme=theme, n=n, doc=doc, topic=topic)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    if q is None:
        raise HTTPException(404, "No matching course material for the given selection.")
    return q


@app.post("/api/quiz/grade")
def grade_quiz(payload: dict):
    """Grade submitted answers -> score + explanations. Send no answers to reveal."""
    quiz_id = payload.get("quiz_id")
    answers = payload.get("answers") or {}
    result = quiz.grade_quiz(quiz_id, answers=answers)
    if result is None:
        raise HTTPException(404, "Quiz not found (server restarted or expired).")
    return result


# Serve other static assets (css/js) if present
os.makedirs(STATIC_DIR, exist_ok=True)
