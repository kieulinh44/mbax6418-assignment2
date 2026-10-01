"""Unit tests: ingest + chunking primitives (no models, no server)."""
import pytest

from app.hybrid import _chunk_pages, _is_image_query
from app.ingest import _clean, _slug


def test_slug_normalizes_filenames():
    assert _slug("MBAX 6418 - Week 5 - RAG v1.pptx") == "mbax-6418-week-5-rag-v1"
    assert _slug("  Weird___Name!!.PDF ") == "weird-name"
    assert _slug("....") == "doc"


def test_clean_collapses_whitespace_and_blank_lines():
    # tabs/multi-spaces collapse; a space before a newline is retained
    assert _clean("  a\t b \n\n\n\n c  ") == "a b \n\n c"
    assert _clean(None) == ""
    assert _clean("") == ""


def test_chunk_pages_deterministic_ids_and_page_index():
    pages = [
        {"doc": "a", "file": "a.pdf", "kind": "pdf", "page": 1, "section": None, "image": None,
         "text": "word " * 200},          # long -> multiple chunks
        {"doc": "a", "file": "a.pdf", "kind": "pdf", "page": 2, "section": None, "image": None,
         "text": "short page"},
        {"doc": "b", "file": "b.md", "kind": "md", "page": 1, "section": None, "image": None,
         "text": ""},                      # image-only style page -> one empty chunk
    ]
    c1 = _chunk_pages(pages)
    c2 = _chunk_pages(pages)
    # deterministic ids
    assert [c["cid"] for c in c1] == [c["cid"] for c in c2]
    # ids encode doc+page+segment
    assert c1[0]["cid"].startswith("a__p1__")
    # page_index resolves to the right position in the pages list
    by_page = {c["page_index"]: c["doc"] for c in c1}
    assert by_page[0] == "a" and by_page[2] == "b"
    # empty-text page still yields a chunk (visual evidence mapping)
    assert any(c["doc"] == "b" and c["text"] == "" for c in c1)


def test_image_query_detection():
    assert _is_image_query("describe the diagram on slide 10")
    assert _is_image_query("what does this chart show?")
    assert _is_image_query("explain the meme on this slide")
    assert not _is_image_query("what is retrieval augmented generation?")
    assert not _is_image_query("how is attendance graded?")
