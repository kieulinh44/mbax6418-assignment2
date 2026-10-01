"""E2E test for requirement 1a: remove documents through the app.

DELETE /api/documents/{doc} where `doc` is the document SLUG (as reported by
/api/files — e.g. "scratch-remove-me", not "scratch-remove-me.md"). Removing
must purge the file from /api/files AND /api/materials and the index.
"""
import uuid

import pytest

# This test drives real upload+indexing, which needs the heavy model stack.
# Skip (not fail) where only the light test deps are installed (CI).
pytest.importorskip("sentence_transformers")
pytest.importorskip("chromadb")
pytest.importorskip("bm25s")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _unique_md_name():
    return f"scratch-remove-{uuid.uuid4().hex[:8]}.md"


def test_remove_document_via_api():
    fname = _unique_md_name()
    up = client.post("/api/upload",
                     files=[("files", (fname,
                                       b"# Remove me\n\nSCRATCH_REMOVE_TOKEN_4491.",
                                       "text/markdown"))])
    assert up.status_code in (200, 201), up.text

    listed = client.get("/api/files").json()["documents"]
    mine = [d for d in listed if d["doc"] == fname.replace(".md", "")]
    assert mine, f"uploaded doc not listed: {listed}"

    resp = client.delete(f"/api/documents/{mine[0]['doc']}")
    assert resp.status_code in (200, 204), resp.text

    after = client.get("/api/files").json()["documents"]
    assert not any(d["doc"] == mine[0]["doc"] for d in after), \
        "removed document still listed after deletion"

    mats = client.get("/api/materials").json()["materials"]
    assert not any(m["file"] == fname for m in mats), \
        "removed source file still downloadable"

    second = client.delete(f"/api/documents/{mine[0]['doc']}")
    assert second.status_code == 404, "deleting an unknown doc must 404"
