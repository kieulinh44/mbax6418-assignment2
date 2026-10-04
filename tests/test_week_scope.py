"""Offline regressions for query-derived document scope across all indexes."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app import hybrid


def matches(row, where):
    if not where:
        return True
    if "$and" in where:
        return all(matches(row, part) for part in where["$and"])
    return all(row.get(key) in value["$in"] if isinstance(value, dict)
               else row.get(key) == value for key, value in where.items())


@pytest.fixture
def scoped_corpus(monkeypatch):
    corpus = hybrid.Corpus.__new__(hybrid.Corpus)
    corpus.pages = [
        {"doc": "week5", "file": "Week 5.pptx", "page": i + 1,
         "text": "opaque", "image": "fake.png", "kind": "pptx"}
        for i in range(125)
    ] + [
        {"doc": "syllabus", "file": "Syllabus.pdf", "page": 1,
         "text": "week 2 memes", "image": "fake.png", "kind": "pdf"},
        {"doc": "week2", "file": "Week 02.pptx", "page": 33,
         "text": "opaque", "image": "fake.png", "kind": "pptx"},
    ]
    corpus.chunks = [{**p, "cid": f"{p['doc']}__p{p['page']}__0",
                      "page_index": i, "section": None}
                     for i, p in enumerate(corpus.pages)]
    corpus.retriever = Mock()
    corpus.text_enc = Mock()
    corpus.text_enc.encode.return_value = np.array([0.1, 0.2])
    clip = Mock()
    clip.encode.return_value = np.array([0.1, 0.2])
    monkeypatch.setattr(hybrid, "_clip_model", lambda: clip)
    corpus.text_coll = Mock()
    corpus.visual_coll = Mock()
    corpus.visual_coll.count.return_value = len(corpus.pages)
    return corpus


def configure_stage(corpus, stage):
    def keyword(tokens, k):
        return SimpleNamespace(documents=[list(range(k))], scores=[[1.0] * k])

    def vector(rows, id_fn, **kwargs):
        selected = [r for r in rows if matches(r, kwargs.get("where"))]
        selected = selected[:kwargs["n_results"]]
        return {"ids": [[id_fn(r) for r in selected]],
                "distances": [[0.5] * len(selected)]}

    corpus.retriever.retrieve.side_effect = (keyword if stage == "keyword" else
        lambda *a, **kw: SimpleNamespace(documents=[[]], scores=[[]]))
    corpus.text_coll.query.side_effect = lambda **kw: (
        vector(corpus.chunks, lambda r: r["cid"], **kw) if stage == "text"
        else {"ids": [[]], "distances": [[]]})
    corpus.visual_coll.query.side_effect = lambda **kw: (
        vector(corpus.pages, hybrid._visual_id, **kw) if stage == "visual"
        else {"ids": [[]], "distances": [[]]})


@pytest.mark.parametrize("stage", ["keyword", "text", "visual"])
def test_week2_meme_scope_survives_global_cutoffs(scoped_corpus, stage):
    configure_stage(scoped_corpus, stage)
    hits = scoped_corpus._retrieve_unlocked("find all the memes from the week 2 slides",
                                          top_k=10)
    assert [(h["page"]["doc"], h["page"]["page"]) for h in hits] == [("week2", 33)]
    assert scoped_corpus.retriever.retrieve.call_args.kwargs["k"] == 127
    assert scoped_corpus.text_coll.query.call_args.kwargs["where"] == {"doc": "week2"}
    assert scoped_corpus.visual_coll.query.call_args.kwargs["where"] == {"doc": "week2"}


@pytest.mark.parametrize("query", ["week2", "Week 02", "week two", "W2",
                                   "weeks 2 and 5", "weeks two, five"])
def test_week_reference_forms(query):
    pages = [{"doc": "opaque-slug", "file": "MBAX_Week_02_LLM.pptx"},
             {"doc": "week-5", "file": "Lecture.pptx"},
             {"doc": "week20", "file": "Week 20.pptx"},
             {"doc": "syllabus", "file": "Syllabus.pdf", "text": "week2"}]
    expected = {"opaque-slug", "week-5"} if query.startswith("weeks") else {"opaque-slug"}
    assert hybrid.query_document_scope(query, pages) == expected


@pytest.mark.parametrize("query,doc,expected", [
    ("Explain transformers", None, None),
    ("Explain transformers", "week2", {"week2"}),
    ("Explain transformers", "absent", set()),
    ("Week 99 memes", None, set()),
    ("week nine", None, set()),
    ("week 20", None, {"week20"}),
    ("Compare week2 and week5", None, {"week2", "week5"}),
    ("Compare week2 and week99", None, {"week2"}),
    ("Week 2 memes", "week5", set()),
    ("Compare week2 and week5", "week5", {"week5"}),
    ("Explain slide 2", None, None),
])
def test_scope_contract(query, doc, expected):
    pages = [{"doc": f"week{n}", "file": f"Week {n}.pptx"} for n in (2, 5, 20)]
    pages.append({"doc": "syllabus", "file": "Syllabus.pdf", "text": "week 2 week 99"})
    before = [dict(p) for p in pages]
    assert hybrid.query_document_scope(query, pages, doc) == expected
    assert pages == before


@pytest.mark.parametrize("query,doc", [("Week 99 memes", None),
                                       ("Week 2 memes", "week5")])
def test_empty_scope_does_not_search_unrelated_sources(scoped_corpus, query, doc):
    assert scoped_corpus._retrieve_unlocked(query, doc=doc) == []
    scoped_corpus.retriever.retrieve.assert_not_called()
    scoped_corpus.text_enc.encode.assert_not_called()
    scoped_corpus.text_coll.query.assert_not_called()
    scoped_corpus.visual_coll.query.assert_not_called()


@pytest.mark.parametrize("query,doc,where", [
    ("Compare week 2 and week 5 slide 33 image", None,
     {"$and": [{"doc": {"$in": ["week2", "week5"]}}, {"page": 33}]}),
    ("Compare week 2 and week 5 slide 33 image", "week2",
     {"$and": [{"doc": "week2"}, {"page": 33}]}),
    ("week 2 slide 33 image", None,
     {"$and": [{"doc": "week2"}, {"page": 33}]}),
    ("Compare week 2 and week 5 images", None,
     {"doc": {"$in": ["week2", "week5"]}}),
    ("week 2 slide 1 and 33 image", None, {"doc": "week2"}),
    ("ordinary image query", None, None),
    ("ordinary image query", "week5", {"doc": "week5"}),
])
def test_vector_scope_and_page_constraints(scoped_corpus, query, doc, where):
    configure_stage(scoped_corpus, "visual")
    hits = scoped_corpus._retrieve_unlocked(query, doc=doc, top_k=200)
    assert hits
    assert scoped_corpus.text_coll.query.call_args.kwargs["where"] == where
    assert scoped_corpus.visual_coll.query.call_args.kwargs["where"] == where
    if "week 2" in query and "week 5" not in query:
        assert {h["page"]["doc"] for h in hits} == {"week2"}


def test_same_week_preserves_all_sources_and_shorthand_identities():
    pages = [{"doc": "notes-w2", "file": "Notes.pdf"},
             {"doc": "slides", "file": "Week two.pptx"},
             {"doc": "week20", "file": "Week 20.pptx"}]
    assert hybrid.query_document_scope("week 2", pages) == {"notes-w2", "slides"}
    assert hybrid.query_document_scope("week 2", []) == set()
    assert hybrid.query_document_scope("ordinary question", []) is None


def test_ordinary_query_keeps_global_keyword_limit(scoped_corpus):
    configure_stage(scoped_corpus, "keyword")
    scoped_corpus._retrieve_unlocked("ordinary query")
    assert scoped_corpus.retriever.retrieve.call_args.kwargs["k"] == 100
    assert scoped_corpus.text_coll.query.call_args.kwargs["where"] is None


def test_week_scope_intersects_topic_and_text_only_mode(scoped_corpus):
    configure_stage(scoped_corpus, "keyword")
    assert scoped_corpus._retrieve_unlocked("week2 memes", topic="missing") == []
    scoped_corpus.retriever.retrieve.assert_not_called()
    hits = scoped_corpus._retrieve_unlocked("week2 memes", topic="opaque",
                                          retrieval_mode="text_keyword_only")
    assert {h["page"]["doc"] for h in hits} == {"week2"}
    scoped_corpus.visual_coll.query.assert_not_called()
