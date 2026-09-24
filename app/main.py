"""FastAPI app: ingest-facing file list + grounded Q&A endpoint."""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config, llm, retrieval

app = FastAPI(title="Course Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    pages = retrieval.load_pages()
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


def _cite(page):
    return {
        "doc": page["doc"],
        "file": page["file"],
        "page": page["page"],
        "section": page.get("section"),
        "excerpt": (page.get("text") or "")[:500],
        "image": f"/pages/{os.path.basename(page['image'])}" if page.get("image") else None,
    }


@app.post("/api/ask")
def ask(payload: dict):
    question = (payload.get("question") or "").strip()
    doc = payload.get("doc") or None
    topic = payload.get("topic") or None
    if not question:
        raise HTTPException(400, "question is required")

    hits = retrieval.retrieve(question, doc=doc, topic=topic)
    if not hits:
        return {
            "answer": "I could not find any matching material in the course files. "
                      "If this topic is covered, please ensure the documents are "
                      "ingested (run ingest) and check the file/filter selection.",
            "sources": [],
            "vision_notes": "",
        }

    # Build context from retrieved pages; keep the full cited excerpt
    context_blocks = []
    for h in hits:
        p = h["page"]
        context_blocks.append(
            f"[Document: {p['file']} | page/slide {p['page']}"
            + (f" | section: {p['section']}" if p.get("section") else "")
            + "]\n" + (p.get("text") or "").strip()
        )
    context = "\n\n".join(context_blocks)

    # If the top page has an image, ask the vision model to read any diagram/chart
    vision_notes = ""
    top = hits[0]["page"]
    if top.get("image"):
        img_path = os.path.join(config.DATA_INDEX, top["image"])
        if os.path.exists(img_path):
            try:
                vision_notes = llm.vision(
                    "Describe exactly what this page/slide image shows — especially any "
                    "diagram, chart, graph, or formula. Be concise and factual. If the "
                    "image is mostly text or blank, say so.",
                    [img_path],
                )
            except Exception:
                vision_notes = ""

    sys = (
        "You are a course assistant. Answer ONLY from the provided course material "
        "context. Every factual point must be grounded in the cited pages. Cite each "
        "source as [doc: <file> p.<page>] where you use it. If the material does not "
        "contain the answer, say clearly that the information is not in the materials "
        "and DO NOT invent facts or citations. Below, an image description of a top "
        "diagram is included where available."
    )
    user = (f"QUESTION: {question}\n\n" + ("VISION NOTES (top page diagram):\n" + vision_notes + "\n\n" if vision_notes else "")
            + "COURSE MATERIAL:\n" + context
            + "\n\nAnswer concisely, cite sources inline as [file p.N], and end with a "
              "'Sources:' section listing the file, page/slide, and a short reason each is relevant.")

    try:
        answer = llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {e}")

    return {
        "answer": answer,
        "sources": [_cite(h["page"]) for h in hits],
        "vision_notes": vision_notes,
    }


# Serve other static assets (css/js) if present
os.makedirs(STATIC_DIR, exist_ok=True)
