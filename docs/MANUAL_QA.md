# Manual QA Checklist — MBAX 6418 Assignment 2

How to verify the app by hand (assignment 4a). Fill in **Result** with
PASS / FAIL / BLOCKED / PENDING and the **date + tester**. Sources of truth:
the app running locally (`python -m uvicorn app.main:app --port 8000`) with
`.env` configured, real course materials in `data/raw/`, and
`python -m scripts.ingest` after every material change.

Quick start for a clean check:
```bash
pip install -r requirements.txt            # full model stack
cp .env.example .env                       # then fill in REAL values from the team
python scripts/sample_pdf.py               # demo material (no Canvas files needed)
python -m scripts.ingest                   # build indexes (needs .env? no — models local)
python -m uvicorn app.main:app --port 8000
# open http://127.0.0.1:8000
```

---

## 1a — Manage course materials

| # | Check | How | Result | Notes |
|---|---|---|---|---|
| 1.1 | Upload a document through the app | Materials card → choose PDF → Upload & index; confirm the file appears in the file list | | |
| 1.2 | Upload same file twice → no duplicate | Upload the same file again; expect a clear "already indexed" message and no second copy | | |
| 1.3 | Renamed re-upload (content same, name different) | Copy file to a new name, upload; note whether content now appears twice (known weakness: dedup is by filename slug only) | | |
| 1.4 | Remove a document through the app | **BLOCKED — feature not implemented yet** (see `tests/test_remove_document_e2e.py`); verify after teammates land it: delete → file gone from list → answers no longer cite it | BLOCKED | req 1a |
| 1.5 | Unsupported format rejected | Try uploading `.zip`/`.exe`; expect 400 with the accepted-format list | | |
| 1.6 | PPTX renders as slide images | Upload a PPTX with LibreOffice installed; ask a slide question; confirm the answer shows the *actual slide* image | | needs LibreOffice |
| 1.7 | Download a material file | Click a download link; file arrives intact (compare checksum) | | |

## 2a — Retrieve / show / explain visual content

| # | Check | How | Result | Notes |
|---|---|---|---|---|
| 2.1 | Slide images shown with doc + slide number | Ask a question about the slides; verify each source shows the document name, slide/page number, and the real slide image | | |
| 2.2 | Describe an image/diagram | Ask e.g. "what does the chart on slide 12 show?"; verify the answer uses the vision pass and the displayed image supports it | | needs vision endpoint in `.env` |
| 2.3 | **Meme test**: "Vibe Coding on Prod" | Ask the app to find the meme in the Week 2 slides; it must retrieve + display the correct slide and summarize it | PENDING | needs real Week 2 slides in `data/raw` |
| 2.4 | Source matches the original slide | For a sample answer, open the original PDF/PPTX and confirm the cited page/slide really contains the claim | | |
| 2.5 | No-match honesty | Ask something impossible ("quantum chromodynamics"); expect an honest "not in the materials" answer, no invented citation | | covered by automated tests too |

## 3a — Quiz generation and feedback

| # | Check | How | Result | Notes |
|---|---|---|---|---|
| 3.1 | Questions answerable from the selected material | Generate a quiz on a doc/topic you know; every question must be answerable from that material | | |
| 3.2 | Answer key stays hidden | Inspect the quiz API response (Browser devtools): no `answer_idx` / `explanation` before answering | | covered by `test_quiz_key_hiding.py` |
| 3.3 | Score matches the key | Answer some questions correctly/intentionally wrong; confirm score + per-question verdicts match the key | | |
| 3.4 | Explanations + sources shown after submit | After grading, each question shows an explanation and a source (file, page, excerpt/image) | | |
| 3.5 | Reveal on demand | Submit with no answers; solutions should be returned for review | | |

## 4a — Robustness / edge cases

| # | Check | How | Result | Notes |
|---|---|---|---|---|
| 4.1 | Model services down | Stop `.env` endpoints (or unset keys); ask + quiz → graceful `service_unavailable` / 503 messages, no crash, no raw exception | | covered by `test_unavailable_services.py` |
| 4.2 | Missing information | Ask an out-of-scope question; expect honest refusal (covered) | | |
| 4.3 | Repeated uploads + re-ingestion | Upload twice, remove `data/index`, re-ingest; app recovers cleanly | | |
| 4.4 | Light and dark modes | Current UI is light-only; check all text/controls readable; note if dark mode is needed for the "if available" requirement | | no dark theme implemented |
| 4.5 | Fresh-clone setup | On a clean machine: README steps → app runs; note any missing step (assignment: fix what you discover) | | |
| 4.6 | Screenshot content verified | `docs/screenshots/qa.jpg` + `quiz.jpg` match the current app UI and show (a) answer with slide image, (b) quiz feedback with sources; re-capture if outdated | PENDING | no vision check run yet |

---

## Recorded results — E2E run of `mirina-debug` (2026-09-30, local, no `.env`)

Full pipeline verified by actually running the app (uvicorn, port 8010) with
real models (MiniLM + CLIP + chromadb + bm25s) over `scripts/sample_pdf.py`:

| Check | Result | Evidence |
|---|---|---|
| 1.2 same file twice | **PASS** — HTTP 409 "Already indexed (rename or delete first)" | curl re-upload |
| 1.5 unsupported format | **PASS** — HTTP 400 with accepted-format list (`.docx, .md, .pdf, .pptx, .txt`) | zip upload |
| 2.1 slide/page evidence | **PASS (API)** — ask returned sources with doc + page + excerpt + `/pages/…` image URLs; retrieval found the right page (L2 regularization, p.2) | `/api/ask` response |
| 4.1 model services down | **PASS** — ask → 200, `service_unavailable: true`, honest message + closest material with citation; quiz → HTTP 503 "model service unavailable" (no 404, no crash) | `/api/ask`, `/api/quiz` |
| 2.5 no-match honesty (model down) | **CONDITIONAL** — honest "could not find any matching material" only when retrieval returns zero hits; with a loose hit, the fallback surfaces "closest material" (acceptable, flagged as a weakness to revisit) | `/api/ask` out-of-scope probe |
| 1.7 download | PENDING (route exists, `basename`-guarded) | — |
| 4.4 dark mode | NOT IMPLEMENTED (light theme only; requirement is conditional "if available") | `static/index.html` |
| Pending | UI-level checks (buttons, download, quiz in browser), meme test with real Week 2 slides, screenshot content verification | needs human + materials |

## Known gaps this checklist exposes (as of Sep 2026)

- **Remove-document (1a)** — not implemented; only the acceptance test exists.
- **Dedup** — by filename slug, not content hash (rename ⇒ duplicate).
- **Dark mode** — not implemented (requirement is conditional: "if available").
- **Meme test + real Week 2 slides** — needs the actual Canvas materials.
- **Screenshots content** — never visually verified against the running app.
