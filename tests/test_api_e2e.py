"""End-to-end API tests against a running app instance (http://127.0.0.1:8000).

Skipped automatically when the server is not reachable. The upload/delete test
cleans up after itself so the shared local index is left unchanged.
"""
import os
import uuid

import pytest
import requests

BASE = os.environ.get("CA_BASE", "http://127.0.0.1:8000")


def _up():
    try:
        return requests.get(f"{BASE}/api/health", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason=f"app not running at {BASE}")


def test_health_and_home():
    assert requests.get(f"{BASE}/api/health", timeout=10).json()["ok"] is True
    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200 and "Course Assistant" in r.text


def test_files_and_materials():
    files = requests.get(f"{BASE}/api/files", timeout=10).json()
    mats = requests.get(f"{BASE}/api/materials", timeout=10).json()
    assert files["documents"], "expected at least one indexed document"
    names = {m["file"] for m in mats["materials"]}
    for d in files["documents"]:
        assert d["doc"], "document missing slug"
    assert names, "expected at least one downloadable material"


def test_ask_returns_answer_and_sources():
    r = requests.post(f"{BASE}/api/ask", json={"question": "What is RAG?"}, timeout=240)
    assert r.status_code == 200
    d = r.json()
    assert d["answer"] and "sources" in d
    assert all("file" in s and "page" in s and "excerpt" in s for s in d["sources"])


def test_upload_dedupe_and_delete_cycle():
    tag = uuid.uuid4().hex[:6]
    fname = f"e2e_{tag}.md"
    content = f"# E2E test doc {tag}\nMLOps model registries store versions and metadata.\n"
    try:
        # 1) upload
        r = requests.post(f"{BASE}/api/upload", files={"files": (fname, content)},
                          timeout=300)
        assert r.status_code == 200, r.text
        doc = f"e2e-{tag}"
        files = requests.get(f"{BASE}/api/files", timeout=10).json()
        assert any(d["doc"] == doc for d in files["documents"])
        # 2) same file again -> 409, no duplicate
        r2 = requests.post(f"{BASE}/api/upload", files={"files": (fname, content)},
                           timeout=60)
        assert r2.status_code == 409, "duplicate upload should be refused"
        pages_before = {d["doc"]: d for d in requests.get(f"{BASE}/api/files",
                                                          timeout=10).json()["documents"]}
        assert pages_before[doc]["pages"] >= 1
        # 3) removal purges it
        rd = requests.delete(f"{BASE}/api/documents/{doc}", timeout=60)
        assert rd.status_code == 200, rd.text
        files2 = requests.get(f"{BASE}/api/files", timeout=10).json()
        assert all(d["doc"] != doc for d in files2["documents"])
        mats2 = {m["file"] for m in requests.get(f"{BASE}/api/materials",
                                                 timeout=10).json()["materials"]}
        assert fname not in mats2, "raw file should be gone after removal"
    finally:
        requests.delete(f"{BASE}/api/documents/" + ("e2e-" + tag), timeout=30)
