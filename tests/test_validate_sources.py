"""Source-validation unit tests (requirement: check that sources support the answer).

Documents the current behaviour of `app.main._validate` so improvements are
test-pinned: today a source counts as supported only when its FILENAME appears
in the answer text. This is a weak check (no page-level verification) — see
README / debug backlog.
"""
from app.main import _validate

HITS = [
    {"page": {"file": "lecture1.pdf", "page": 2}},
    {"page": {"file": "slides-week2.pptx", "page": 5}},
]


def test_source_mentioned_by_filename_is_supported():
    v = _validate("lecture1.pdf explains overfitting.", HITS)
    assert v["all_sources_supported"] is False  # second source not cited
    assert {"file": "lecture1.pdf", "page": 2} in v["sources_used"]
    assert {"file": "slides-week2.pptx", "page": 5} in v["sources_not_cited"]


def test_no_filename_mentioned_means_nothing_supported():
    v = _validate("Overfitting is when a model memorizes training data.", HITS)
    assert v["all_sources_supported"] is False
    assert v["sources_used"] == []


def test_all_sources_supported_when_all_files_named():
    v = _validate("lecture1.pdf and slides-week2.pptx both cover this.", HITS)
    assert v["all_sources_supported"] is True
