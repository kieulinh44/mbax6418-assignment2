# MBAX 6418 — Assignment 2: Course Assistant

A course assistant that **answers questions** and **generates practice quizzes**
using course materials (slides, syllabus, and other files from Canvas).

**Status:** hybrid-RAG Q&A **and** practice-quiz generator, both live.

## Screenshots

| Q&A (ask + download course materials) | Practice quiz |
|----------------------------------------|---------------|
| ![Q&A dashboard](docs/screenshots/qa.jpg) | ![Practice quiz](docs/screenshots/quiz.jpg) |

## What the app accepts
| Format | Text | Page/slide image |
|--------|------|------------------|
| `.pdf` | ✅ | ✅ rendered PNG |
| `.pptx` | ✅ | ✅ rendered PNG (requires LibreOffice) |
| `.docx` | ✅ | ⏳ phase 2 |
| `.md` / `.txt` | ✅ | n/a |

From the dashboard, students can **download the source files** or **upload new
material** (PDF/PPTX/DOCX/MD/TXT); uploads are ingested and indexed
incrementally without restarting the server.

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

**Hybrid RAG** over the course materials:

```
course files → ingest (text + rendered page/slide image per page)
            → chunk pages (text splitter) preserving doc·page·section
            → three separate indexes in chromadb + bm25s:
                keyword (BM25)  ·  text embeddings (all-MiniLM)  ·  visual (CLIP images)
query → run all three → reciprocal-rank fusion (rerank)
     → top text chunks + their page images as evidence
     → grounded answer (DeepSeek) + vision pass (Qwen) over relevant images
     → {answer, sources, validation} — sources validated against evidence
```

- **Chunking:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter`), each chunk tagged with doc/file/page/section.
- **Keyword:** `bm25s`; **text vectors:** `chromadb` collection with all-MiniLM; **visual vectors:** `chromadb` collection with CLIP `clip-ViT-B-32` over every page image, queried cross-modally by the text question.
- **Combine/rerank:** Reciprocal Rank Fusion across the three ranked lists, collapsed to pages keeping each page's best chunk + original image.
- **Validation:** after generation, citations are checked against the retrieved evidence; the response carries separate `answer` and `sources` fields plus `validation.all_sources_supported`.

## Notes
* Course material and the generated index are **gitignored** (public repo;
  Canvas content may be copyrighted). Run ingest locally.
* Model endpoints are OpenAI-compatible and configured via environment
  variables. Copy `.env.example` to `.env`, fill in the real values the team
  provides, and never commit `.env`. Without these variables the app starts
  but model calls (ask/quiz) fail — see Setup.

## Tests
```bash
pip install -r requirements-dev.txt   # pytest + httpx (test-only deps)
pytest -q                              # fast, offline unit + API tests
python scripts/scan_secrets.py         # fail if a real key/endpoint leaks
```
`tests/test_remove_document_e2e.py` is expected to fail (xfail) until the
remove-document feature from requirement 1a is implemented.
