# MBAX 6418 — Assignment 2: Course Assistant

A course assistant that **answers questions** and **generates practice quizzes**
using course materials (slides, syllabus, and other files from Canvas).

**Status: phase 1 skeleton** — ingest + grounded Q&A with sources and page images.
Quizzes are the next phase.

## What the app accepts
| Format | Text | Page/slide image |
|--------|------|------------------|
| `.pdf` | ✅ | ✅ rendered PNG |
| `.pptx` | ✅ | ✅ rendered PNG (requires LibreOffice) |
| `.docx` | ✅ | ⏳ phase 2 |
| `.md` / `.txt` | ✅ | n/a |

The front page documents this list.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended: PPTX slides render as images only if LibreOffice is
# installed (winget install --id TheDocumentFoundation.LibreOffice -e).
```

## Run
```bash
# 1. Put your course files in data/raw/ (PDFs, slides, syllabus, notes)
# 2. Build the index:
python -m scripts.ingest
# 3. Start the app:
python -m uvicorn app.main:app --reload --port 8000
# 4. Open http://localhost:8000
```

## Test sample (no real materials needed)
```bash
python scripts/sample_pdf.py   # writes data/raw/sample-lecture.pdf
python -m scripts.ingest
python -m uvicorn app.main:app --reload --port 8000
```
Then ask: "What is L2 regularization and how does it differ from L1?"

## Architecture
COURSE FILES → ingest (text + rendered page image per page) → vector index
→ retrieve(question, [doc filter], [topic filter]) → grounded answer via the
local reasoning model, with a vision pass over the top page's image to read
diagrams/charts → cited sources + original page images shown in the browser.

## Notes
- Course material and the generated index are **gitignored** (public repo;
  Canvas content may be copyrighted). Run ingest locally.
- Model endpoints are OpenAI-compatible and configured in `app/config.py`
  (override keys via env vars; defaults point at the class's local endpoints).
