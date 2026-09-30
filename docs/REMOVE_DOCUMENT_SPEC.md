# Remove-Document Feature — Implementation Spec (requirement 1a)

**Status:** PENDING — not implemented. Acceptance test exists (expected-fail):
`tests/test_remove_document_e2e.py` — make it pass and the feature is done.

## Why

Assignment 2, section 1a: "Users must be able to add **and remove** documents
through the app. … When a document is removed, remove its searchable content
too, so later answers do not rely on it."

## Acceptance criteria (from the xfail test)

1. `DELETE /api/documents/{filename}` returns a 2xx for a known document.
2. After deletion, the document no longer appears in `GET /api/files`.
3. (For completeness) `GET /api/materials` and upload-level duplicate checks
   must also stop seeing it, and answers must never cite it again.

## What has to change

- **Backend** (`app/main.py`): a `DELETE /api/documents/{fname}` route
  (path-traversal safe, like `/api/download` — use `os.path.basename`).
- **Index purge** (`app/hybrid.py`): remove the doc's pages from
  `data/index/pages.json` and `chunks.json`, delete the doc's page images from
  `data/index/pages/`, remove its chunk ids from the chroma `text` collection
  and page ids from `visual`, and rebuild the in-memory BM25 retriever
  (`_build_keyword`) for the remaining chunks.
- **Raw file**: also delete/move `data/raw/{filename}` so the "already
  indexed" duplicate check and the download list stay consistent.
- **Frontend** (`static/index.html`): a per-document "remove" button in the
  materials/file list that calls the DELETE endpoint and refreshes.

## Guardrails

- Never delete arbitrary paths: only files that are currently in the index,
  resolved through `pages.json` (source of truth), never a raw user-supplied
  path.
- Concurrency: reuse `Corpus._lock` (ingest holds it; deletion must too).
- If the doc is mid-upload/quiz, behave sanely (return 409 or wait on the lock).
- Keep it server-side consistent: update `pages.json` + `chunks.json` + chroma
  + BM25 + raw file, or none (transactional error handling).

## Tests once implemented

- Flip `tests/test_remove_document_e2e.py` from xfail to a normal test.
- Add: deleting an unknown file → 404; deleting then asking a question that
  only that file answered → honest "not in materials" (missing-info path),
  never a stale citation.
