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

From the dashboard, students can **download the source files**, **upload new
material** (PDF/PPTX/DOCX/MD/TXT), and **remove** documents (removal purges the
document's searchable content — pages, slide images, embeddings — so later
answers don't rely on it). Uploading the same file twice is refused (no
duplicates).

## PowerPoint (.pptx) conversion
Direct upload of `.pptx` decks is supported. Under the hood the app uses
**Python + LibreOffice** to convert each deck to PDF and render every slide to
an image (so answers can show pictures of the actual slides). 

- **Required extra software:** LibreOffice
  (`winget install --id TheDocumentFoundation.LibreOffice -e`, or install from
  libreoffice.org). Without it, `.pptx` text still indexes but slide *images*
  are not produced.
- **How conversion works:** upload → LibreOffice `--convert-to pdf` → each PDF
  page rendered as a PNG at dpi 120 → image stored alongside the slide text.
- **Manual workaround:** exporting a deck to PDF yourself (PowerPoint: File →
  Export → PDF) and uploading the PDF works equally well and needs no extra
  software.
- **Verify converted slides look correct:** after uploading, find the deck in
  the file list, ask the assistant a question about it, and confirm the slide
  image shown in the sources matches the deck's slide (title text, diagrams,
  and layout intact). Slide images are also listed under
  `data/index/pages/` (files named `<deck>_sNNN.png` — open a few and eyeball
  them against the original deck).

## Slide-image RAG and visual questions

Q&A combines BM25 keyword retrieval, text embeddings, and CLIP slide-image
embeddings. Retrieved source images are rendered copies of the original slides,
not AI-generated illustrations. The answer appears next to a gallery with the
**document filename and Slide N / Page N**, full-size image links, and an
**Explain this slide** action. On smaller screens the gallery stacks below the answer.

Select a course document and try:
- “Explain the diagram showing keyword search, vector search, and reranking.”
- “Explain the two charts on slide 7.”
- “Describe the picture on slide 33 and explain its message.”

A singular slide/page number constrains retrieval within the selected document.
Ambiguous ranges or multiple slide references retain the ordinary hybrid ranking.
The vision model analyzes up to three available retrieved images in parallel,
with the question and exact source label supplied separately for each image.
Its attributed observations are passed to the answer model as course evidence,
including information missing from extracted text. For retrieved PowerPoint slides,
the vision pass uses up to three native-resolution embedded pictures from
that exact slide, cropped to the visible picture area; tiny logos are omitted.
The full-slide render is analyzed first to preserve layout and spatial context;
the native details then improve small-label legibility. Slides without usable
raster details use the full-slide render alone. The displayed evidence remains
the actual full slide in either case.
Retrieval remains broad enough to compare candidate evidence internally, but the
dashboard gallery shows only sources cited by the validated answer. A singular
visual-locator request such as “Find the meme about Vibe Coding on \"Prod\"”
shows only the highest-ranked supporting slide; comparison questions can still
show multiple cited slides.
Failed image analysis or missing renders produce explicit limitations instead of
unsupported visual claims. Internal reasoning and truncated model completions
are never presented as finished answers. Before display, a separate deterministic
grounding review revises unsupported claims or rejects the draft if validation
cannot complete. Visual answers separate direct observations from interpretation,
and exact labels or values are allowed only when the evidence marks them legible.
The API returns `vision_sources` and `visual_warnings` alongside `answer`,
`sources` (with `kind`, `label`, and an existing image URL), and `vision_notes`.

**PPTX rendering is required for visual questions:**
```bash
# macOS
brew install --cask libreoffice
# After installing it, rebuild any previously text-only slide index:
python -m scripts.ingest
```
PowerPoint conversion uses a separate headless LibreOffice profile, includes
hidden slides to preserve original numbering, and checks the rendered slide count.
PDF exports can also be uploaded when LibreOffice is unavailable. Original content
and generated indexes remain local and gitignored.

### Regression tests
```bash
# Development dependency only; the app does not require Playwright at runtime.
pip install playwright
python -m playwright install chromium
# Start the local app on port 8000 before running browser tests.
python -m unittest discover -s tests -v
```
The visual-answer and retrieval unit tests isolate external model boundaries.
Live checks must additionally confirm that actual PNGs load and that model
explanations match the retrieved course slides.

## Test question set
A graded set of 5–10 questions (syllabus, slide text, **visual questions
including the meme slide**, and one intentionally unanswerable question) lives
in [docs/EVALUATION.md](docs/EVALUATION.md), with a runner script:
```bash
python scripts/eval_questions.py   # runs each question against the local app
```

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

## Quick start (real materials)
```bash
# 1. Put course files in data/raw/ (or upload them from the dashboard once running)
python -m scripts.ingest          # ingest + build the hybrid index
python -m uvicorn app.main:app --port 8000
# open http://localhost:8000 — Q&A, quizzes, download/upload
```

## Architecture

**Hybrid RAG** over the course materials:

```
course files → ingest (text + rendered page/slide image per page)
            → chunk pages (text splitter) preserving doc·page·section
            → three separate indexes in chromadb + bm25s:
                keyword (BM25)  ·  text embeddings (all-MiniLM)  ·  visual (CLIP images)
query → run all three → weighted score fusion (rerank)
     → top text chunks + their page images as evidence
     → grounded answer (DeepSeek) + vision pass (Qwen) over relevant images
     → {answer, sources, validation} — sources validated against evidence
```

- **Chunking:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter`), each chunk tagged with doc/file/page/section (deterministic ids so uploads/removals update incrementally).
- **Keyword:** `bm25s`; **text vectors:** `chromadb` collection with all-MiniLM; **visual vectors:** `chromadb` collection with CLIP `clip-ViT-B-32` over every page image, queried cross-modally by the text question.
- **Combine/rerank:** weighted score fusion — per-stage min-max normalization of BM25, text-cosine, and visual-cosine scores, summed with weights (visual boosts surfaced pages for text questions; visual leads when the question is explicitly about an image/diagram/meme).
- **Add/remove:** the dashboard supports upload (incremental indexing) and removal (purges the document's pages, chunks, embeddings, and images); duplicate uploads are refused.
- **Validation:** after generation, a second evidence-only pass revises or rejects unsupported claims, and citations must match both the retrieved filename and exact page/slide number. The response carries separate `answer` and `sources` fields plus `validation.all_sources_supported` and `validation.grounding_review`.

## Notes
- Course material and the generated index are **gitignored** (public repo;
  Canvas content may be copyrighted). Run ingest locally.
- Model endpoints are OpenAI-compatible and configured in `app/config.py`
  (override keys via env vars; defaults point at the class's local endpoints).
