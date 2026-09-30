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

## Evaluation (H1–H3)

Question set (8: 5 text · 2 visual incl. the meme · 1 unanswerable) in
`docs/evaluation/questions.json`; results + interpretation in
`docs/results/evaluation.md`. Re-run with

```bash
python scripts/eval_retrieval.py
```

**Interpretation of the current run:** on the 3-page sample deck all
approaches tie (identical top-1 hits, precision@3 capped at 1/3) — the sample
is too small to separate keyword / vector / hybrid (RRF). The harness is the
deliverable: rerun it on the real course deck, then fill the
answer-correctness + source-support columns (they need `.env` + human
judgment) and update this section. Hybrid adds ~100 ms/query warm (plus a
one-time CLIP load) for the visual evidence it provides.

## Tested vs Unchecked (as of the debug branch)

**Tested ✅** (offline, `pytest -q` → 12 passed, 1 xfailed; run on `mirina-debug`)
- Security config: no committed keys/endpoints; env-only resolution with dummy
  examples (`tests/test_config_secrets.py`)
- Quiz answer key + explanations never sent to the client; grading + reveal
  (`tests/test_quiz_key_hiding.py`)
- Source validation behaviour pinned (`tests/test_validate_sources.py`)
- Unavailable model services: ask degrades gracefully with evidence
  (`service_unavailable`), quiz returns a clear 503 instead of a misleading 404
  (`tests/test_unavailable_services.py`)
- Secrets-leak scan over all tracked files (`python scripts/scan_secrets.py`)

**Unchecked / blocked ⚠️** (see `docs/MANUAL_QA.md` for the full checklist)
- Remove-document feature (requirement 1a) — not implemented; pinned by an
  xfail acceptance test, awaiting implementation.
- Manual QA on real materials: the "Vibe Coding on Prod" meme test (needs the
  actual Week 2 slides), source-vs-original-slide checks.
- Full end-to-end run against the class model endpoints (ask/quiz with real
  chat + vision services) — the automated suite is offline/stubbed by design.
- Dedup is by filename slug only, not content hash (rename ⇒ duplicate).
- Dark mode — not implemented (requirement is conditional: "if available").
- Screenshot content never visually verified against the running app.
- Evaluation deliverable (5-10 question set, two-approach RAG comparison with
  time + answer correctness + source support) and the SVG architecture diagram
  are still outstanding.
