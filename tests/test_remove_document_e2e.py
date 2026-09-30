"""E2E test for requirement 1a: remove documents through the app.

Expected to FAIL (xfail) until the delete endpoint + index purge exist.
The feature is missing on `main` and `mirina-debug`; this test pins the
acceptance criteria so whoever implements it can iterate against it:
  - deleting a document returns 2xx
  - the document disappears from /api/files afterwards
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="remove-document feature (assignment 1a) not implemented yet",
)


def test_remove_document_via_api():
    client = TestClient(app)
    # upload a tiny text file first, then remove it
    up = client.post("/api/upload",
                     files=[("files", ("scratch-remove-me.md",
                                        b"# Remove me\n\nSCRATCH_REMOVE_TOKEN_4491.",
                                        "text/markdown"))])
    assert up.status_code in (200, 201), up.text

    listed = client.get("/api/files").json()["documents"]
    assert any("scratch-remove-me" in d["doc"] for d in listed)

    resp = client.delete("/api/documents/scratch-remove-me.md")
    assert resp.status_code in (200, 204), resp.text

    after = client.get("/api/files").json()["documents"]
    assert not any("scratch-remove-me" in d["doc"] for d in after), \
        "removed document still listed after deletion"
