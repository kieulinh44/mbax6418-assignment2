"""Ingest course materials into per-page "documents".

Each page/slide keeps its extracted TEXT plus a rendered COPY of the original
image (PNG). This satisfies the requirement to preserve both text and the
original page/slide image for evidence display.

Formats handled this phase:
  - PDF      -> text via PyMuPDF + full-page PNG render
  - PPTX/DOCX-> text extraction (page/slide image render needs LibreOffice,
                flagged as phase 2)
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid

import fitz  # PyMuPDF

from . import config

SUPPORTED_EXTS = {".pdf", ".pptx", ".docx", ".md", ".txt"}

# Common LibreOffice locations; used to render PPTX slides to images.
_SOFFICE = None


def _find_soffice():
    global _SOFFICE
    if _SOFFICE is not None:
        return _SOFFICE
    candidates = shutil.which("soffice") or shutil.which("libreoffice") or ""
    if not candidates:
        candidates = [
            r"C:/Program Files/LibreOffice/program/soffice.exe",
            r"C:/Program Files (x86)/LibreOffice/program/soffice.exe",
            "/usr/bin/soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    for c in candidates:
        if c and os.path.exists(c):
            _SOFFICE = c
            return c
    _SOFFICE = None
    return None


def _clean(text):
    if not text:
        return ""
    # collapse repeated blank lines / whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slug(name):
    base = re.sub(r"[^A-Za-z0-9]+", "-", os.path.splitext(name)[0]).strip("-").lower()
    return base or "doc"


def _render_page(doc, page_num, pages_dir, doc_slug):
    """Render a PDF page to PNG; return relative path (or None)."""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=120)
    img_name = f"{doc_slug}_p{page_num + 1:03d}.png"
    img_path = os.path.join(pages_dir, img_name)
    pix.save(img_path)
    return os.path.join("pages", img_name)


_HEADING_RE = re.compile(r"^\s*(#+|[0-9]+[.)]\s+|\*{1,3}\s+)", re.I)


def guess_section(text):
    """Best-effort section heading from the page text."""
    for line in text.splitlines()[:15]:
        s = line.strip()
        if 2 <= len(s) <= 80 and _HEADING_RE.match(s):
            return s.lstrip("#*.) 0-9 ").strip()
        if 2 <= len(s) <= 60 and not s.endswith((".", ",", ";")) and s.isprintable():
            uppercase = sum(1 for c in s if c.isupper()) / max(len(s), 1)
            if uppercase > 0.55:
                return s
    return None


def _ingest_pdf(path):
    doc_id = _slug(os.path.basename(path))
    pages_dir = os.path.join(config.DATA_INDEX, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    pages = []
    with fitz.open(path) as doc:
        for i in range(len(doc)):
            page = doc[i]
            text = _clean(page.get_text())
            img_rel = _render_page(doc, i, pages_dir, doc_id)
            pages.append({
                "doc": doc_id,
                "file": os.path.basename(path),
                "kind": "pdf",
                "page": i + 1,                       # 1-based
                "section": guess_section(text),
                "text": text,
                "image": img_rel,
            })
    return pages


def _ingest_text(path, kind):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [{
        "doc": _slug(os.path.basename(path)),
        "file": os.path.basename(path),
        "kind": kind,
        "page": 1,
        "section": None,
        "text": _clean(text),
        "image": None,
    }]


def _render_pptx_slides(path, doc_id, pages_dir):
    """Render every PPTX slide to a PNG via LibreOffice. Returns {slide_idx: relpath}."""
    soffice = _find_soffice()
    if not soffice:
        return {}
    convert_dir = os.path.join(config.DATA_INDEX, "convert")
    os.makedirs(convert_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.DATA_INDEX) as td:
        pdf_path = os.path.join(td, "deck.pdf")
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", td, path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        pdf = os.path.join(td, "deck.pdf")
        if not os.path.exists(pdf):
            # LibreOffice names output after the source; find the produced pdf
            pdfs = [f for f in os.listdir(td) if f.lower().endswith(".pdf")]
            if pdfs:
                pdf = os.path.join(td, pdfs[0])
        images = {}
        if os.path.exists(pdf):
            with fitz.open(pdf) as doc:
                for i in range(len(doc)):
                    pix = doc[i].get_pixmap(dpi=120)
                    img_name = f"{doc_id}_s{i + 1:03d}.png"
                    pix.save(os.path.join(pages_dir, img_name))
                    images[i + 1] = os.path.join("pages", img_name)  # 1-based slide number
    return images


def _ingest_pptx(path):
    """Text per slide (python-pptx) + rendered slide images (LibreOffice) if available."""
    from pptx import Presentation
    pages_dir = os.path.join(config.DATA_INDEX, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    doc_id = _slug(os.path.basename(path))
    images = _render_pptx_slides(path, doc_id, pages_dir)
    prs = Presentation(path)
    pages = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in para.runs).strip()
                    if t:
                        parts.append(t)
        text = _clean("\n".join(parts))
        if text or i in images:
            pages.append({
                "doc": doc_id,
                "file": os.path.basename(path),
                "kind": "pptx",
                "page": i,
                "section": None,
                "text": text,
                "image": images.get(i),
            })
    return pages


def _ingest_docx(path):
    from docx import Document
    d = Document(path)
    parts = []
    for para in d.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    text = _clean("\n".join(parts))
    return [{
        "doc": _slug(os.path.basename(path)),
        "file": os.path.basename(path),
        "kind": "docx",
        "page": 1,
        "section": None,
        "text": text,
        "image": None,
    }]


def ingest_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _ingest_pdf(path)
    if ext in (".md", ".txt"):
        return _ingest_text(path, ext.lstrip("."))
    if ext == ".pptx":
        return _ingest_pptx(path)
    if ext == ".docx":
        return _ingest_docx(path)
    return []


def ingest_dir(raw_dir=None):
    """Ingest every supported file in the raw dir; write pages.json + PNGs."""
    raw_dir = raw_dir or config.DATA_RAW
    pages_dir = os.path.join(config.DATA_INDEX, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    all_pages = []
    for fname in sorted(os.listdir(raw_dir)):
        fpath = os.path.join(raw_dir, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
            ps = ingest_file(fpath)
            all_pages.extend(ps)
        else:
            print(f"  (skip non-course file: {fname})")
    with open(os.path.join(config.DATA_INDEX, "pages.json"), "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)
    print(f"Ingested {len(all_pages)} page(s) from {raw_dir}")
    print(f"Index written to {config.DATA_INDEX}")
    return all_pages
