# MBAX 6418 — Assignment 2: Course Assistant

A course assistant that **answers questions** and **generates practice quizzes**
using course materials (slides, syllabus, and other files from Canvas).

**Status:** hybrid-RAG Q&A **and** practice-quiz generator, both live.

## What it does

- **Grounded Q&A** — answers from the retrieved course material only, with
  cited sources (document, page/slide, excerpt) and the original slide images
  as visual evidence; never invents facts or citations.
- **Practice quizzes** — multiple-choice questions with a fixed server-side
  answer key, scoring, and explanations that are only revealed after you
  answer or request them.
- **Manage your materials** — upload new course files, download existing ones,
  and remove documents (deletion purges the document's searchable content —
  pages, slide images, embeddings — so later answers don't rely on it).
  Uploading the same file twice is refused (no duplicates).
- **Visual questions** — ask about diagrams, charts, and memes on the slides;
  the app retrieves the actual slide image through RAG and shows it with the
  document name and slide number.
- **Dark mode** — light/dark theme toggle (🌙/☀️); follows your system theme.

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

From the dashboard students can **download the source files**, **upload new
material** (PDF/PPTX/DOCX/MD/TXT), and **remove** documents. Built-in safety:
uploading the same file twice is refused, and removing a document purges its
searchable content (pages, slide images, embeddings) so later answers never
rely on it.

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
  Conversion uses a separate headless LibreOffice profile, includes hidden
  slides to preserve original numbering, and checks the rendered slide count.
- **Manual workaround:** exporting a deck to PDF yourself (PowerPoint: File →
  Export → PDF) and uploading the PDF works equally well and needs no extra
  software. PDF exports can also be uploaded when LibreOffice is unavailable.
- **Verify converted slides look correct:** after uploading, find the deck in
  the file list, ask the assistant a question about it, and confirm the slide
  image shown in the sources matches the deck's slide (title text, diagrams,
  and layout intact). Slide images are also listed under
  `data/index/pages/` (files named `<deck>_sNNN.png` — open a few and eyeball
  them against the original deck).

## Visual Q&A: slide-image RAG, scoped questions, and meme collections

### How visual retrieval works

Q&A combines BM25 keyword retrieval, text embeddings, and CLIP slide-image
embeddings. Retrieved source images are rendered copies of the original slides,
not AI-generated illustrations. The answer appears next to a gallery with the
**document filename and Slide N / Page N**, full-size image links, and an
**Explain this slide** action. On smaller screens the gallery stacks below the
answer.

Select a course document and try:
- "Explain the diagram showing keyword search, vector search, and reranking."
- "Explain the two charts on slide 7."
- "Describe the picture on slide 33 and explain its message."

A singular slide/page number constrains retrieval within the selected document.
Ambiguous ranges or multiple slide references retain the ordinary hybrid
ranking. Retrieval remains broad enough to compare candidate evidence
internally, but the dashboard gallery shows only sources cited by the validated
answer. A singular visual-locator request such as "Find the meme about Vibe
Coding on \"Prod\"" shows only the highest-ranked supporting slide; comparison
questions can still show multiple cited slides.

### Scoped questions

Week references in a question (for example, `week 2`, `Week 02`, or
`week two`) restrict retrieval to matching source filenames/document identities
**before** keyword, text, and visual ranking. A syllabus merely mentioning that
week is not a matching source. The document dropdown and topic filter remain
additional constraints; conflicting or absent scopes return no unrelated
fallback.

### Meme collections ("find all the memes")

Collection requests such as **"find all the memes from the week 2 slides"** use
an exhaustive visual inspection instead of the usual top-six ranked results.
Each eligible rendered slide is classified independently. Only confirmed meme
or humorous-visual matches appear in the source gallery, in slide order.
The answer reports reviewed/total coverage and explicitly marks incomplete
searches when images are missing, classification fails, or a decision is
uncertain. "Complete" means all eligible images received a conclusive model
classification, not a guarantee that model judgments are infallible.

The first collection search may take a few minutes. Conclusive image decisions
are cached locally in `data/index/visual-meme-cache.json` (gitignored); cache
identity includes the image content hash, classifier version, endpoint and
model. Changed images are reclassified. Ordinary single-meme explanations and
standard Q&A continue using hybrid RAG. Collection discovery is only enabled
in `hybrid` mode; `text_keyword_only` evaluation does not silently call vision.

### How the vision pass works

The vision model analyzes up to three available retrieved images in parallel,
with the question and exact source label supplied separately for each image.
Its attributed observations are passed to the answer model as course evidence,
including information missing from extracted text. For retrieved PowerPoint
slides, the vision pass uses up to three native-resolution embedded pictures
from that exact slide, cropped to the visible picture area; tiny logos are
omitted. The full-slide render is analyzed first to preserve layout and spatial
context; the native details then improve small-label legibility. Slides without
usable raster details use the full-slide render alone. The displayed evidence
remains the actual full slide in either case.

Failed image analysis or missing renders produce explicit limitations instead
of unsupported visual claims. Internal reasoning and truncated model
completions are never presented as finished answers. Before display, a separate
deterministic grounding review revises unsupported claims or rejects the draft
if validation cannot complete. Visual answers separate direct observations from
interpretation, and exact labels or values are allowed only when the evidence
marks them legible. The API returns `vision_sources` and `visual_warnings`
alongside `answer`, `sources` (with `kind`, `label`, and an existing image
URL), and `vision_notes`.

**PPTX rendering is required for visual questions:**
```bash
# macOS
brew install --cask libreoffice
# After installing it, rebuild any previously text-only slide index:
python -m scripts.ingest
```
Original content and generated indexes remain local and gitignored.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended: PPTX slides render as images only if LibreOffice is
# installed (winget install --id TheDocumentFoundation.LibreOffice -e).
```

**Model endpoints (chat + vision):** copy `.env.example` to `.env` and fill in
the team's OpenAI-compatible endpoint values, or export the `DOBOLYI_*`
environment variables. The app reads the keys from the environment or the
git-ignored `.env` file at runtime — never from the repository.

## Run

Put course files in `data/raw/` (or upload them from the dashboard once
running), build the index, and start the server:

```bash
python -m scripts.ingest          # ingest + build the hybrid index
python -m uvicorn app.main:app --port 8000
# open http://localhost:8000 — Q&A, quizzes, download/upload
```

**Sample test (no real materials needed):**
```bash
python scripts/sample_pdf.py      # writes data/raw/sample-lecture.pdf
python -m scripts.ingest
python -m uvicorn app.main:app --reload --port 8000
```
Then ask: "What is L2 regularization and how does it differ from L1?"

## Tests

**Regression / browser tests** (Playwright is a development dependency only;
the app does not require it at runtime):

```bash
pip install playwright
python -m playwright install chromium
# Start the local app on port 8000 before running browser tests.
python -m unittest discover -s tests -v
```

The visual-answer and retrieval unit tests isolate external model boundaries.
Live checks must additionally confirm that actual PNGs load and that model
explanations match the retrieved course slides.

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

- **Chunking:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter`),
  each chunk tagged with doc/file/page/section (deterministic ids so
  uploads/removals update incrementally).
- **Keyword:** `bm25s`; **text vectors:** `chromadb` collection with
  all-MiniLM; **visual vectors:** `chromadb` collection with CLIP
  `clip-ViT-B-32` over every page image, queried cross-modally by the text
  question.
- **Combine/rerank:** weighted score fusion — per-stage min-max normalization
  of BM25, text-cosine, and visual-cosine scores, summed with weights (visual
  boosts surfaced pages for text questions; visual leads when the question is
  explicitly about an image/diagram/meme).
- **Add/remove:** the dashboard supports upload (incremental indexing) and
  removal (purges the document's pages, chunks, embeddings, and images);
  duplicate uploads are refused.
- **Validation:** after generation, a second evidence-only pass revises or
  rejects unsupported claims, and citations must match both the retrieved
  filename and exact page/slide number. The response carries separate `answer`
  and `sources` fields plus `validation.all_sources_supported` and
  `validation.grounding_review`.

## Evaluation

The evaluation question set and the comparison harness are the canonical way to
check the assistant against the assignment requirements. Run the questions
with:

```bash
python scripts/eval_questions.py        # runs each question against the local app
python scripts/compare_retrieval.py     # hybrid vs text_keyword_only comparison
```

`compare_retrieval.py` runs every question with the default `hybrid` mode
(BM25 + text embeddings + CLIP visual fusion) and `text_keyword_only` (BM25 +
text embeddings with visual retrieval/fusion disabled). It records timing, raw
answers, sources, validation, and Question 9's missing-information behavior in
`docs/evaluation_results.json`. Correctness and source support require manual
review; the script does not fabricate evaluation outcomes. The full expected
behaviors are in [docs/EVALUATION.md](docs/EVALUATION.md), and the detailed
manual-review report is in [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md).

### Evaluation question set

The same locally indexed syllabus and Week 2–5 slides were used for both
retrieval approaches. The nine evaluation questions were:

1. What is the grading breakdown for MBAX 6418?
2. How is attendance and participation evaluated?
3. What makes a good few-shot example in prompt engineering?
4. What is Retrieval Augmented Generation (RAG), and why is it useful?
5. Why are commits useful when debugging, according to the Week 4 material?
6. What is vibe coding?
7. What does the flowchart on the “What Is RAG?” slide show?
8. Explain the “One Does Not Simply” meme on the Week 2 “Vibe Coding on
   ‘Prod’” slide. Which meme format is it, and what point does it make?
9. What is the professor's favorite programming language?

Questions 1–2 test the syllabus; Questions 3–6 test slide text; Questions 7–8
require visual evidence; and Question 9 is intentionally unanswerable. The
complete expected behavior is also in [docs/EVALUATION.md](docs/EVALUATION.md).

### Evaluation results and interpretation

Using the same locally indexed syllabus and Week 2–5 course slides, both
retrieval modes answered all nine manually reviewed questions correctly with
supported sources. Hybrid retrieval averaged **7.925 seconds**, compared with
**9.719 seconds** for text-and-keyword-only retrieval. Its clearest advantage
was the visual meme question (6.121 seconds versus 22.162 seconds).

We keep **hybrid retrieval** because it retains visual evidence for diagrams
and memes while also performing faster in this evaluation run. The full
per-question comparison, manual source review, timings, and validation notes
are in [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md).

## Notes

- Course material and the generated index are **gitignored** (public repo;
  Canvas content may be copyrighted). Run ingest locally.
- Model endpoints are OpenAI-compatible and configured through environment
  variables or a git-ignored `.env` file (see `.env.example` for the variable
  names with dummy values). No real keys or endpoints are committed.
