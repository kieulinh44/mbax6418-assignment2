import json
from pathlib import Path

import pytest

from app import main
from app.hybrid import Corpus, lexical_match_score, normalize_retrieval_mode
from scripts.compare_retrieval import MODES, QUESTIONS, run_one


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeSession:
    def post(self, url, json, timeout):
        return FakeResponse({
            "answer": "The materials do not contain this information.",
            "sources": [],
            "validation": {"all_sources_supported": True},
        })


class FakeSessionWithAnswer:
    def __init__(self, answer):
        self.answer = answer

    def post(self, url, json, timeout):
        return FakeResponse({
            "answer": self.answer,
            "sources": [],
            "validation": {"all_sources_supported": True},
        })


def test_retrieval_modes_are_validated_and_defaulted():
    assert normalize_retrieval_mode(None) == "hybrid"
    assert normalize_retrieval_mode("text_keyword_only") == "text_keyword_only"
    with pytest.raises(ValueError):
        normalize_retrieval_mode("visual_only")


def test_grading_terms_receive_an_exact_text_match_boost():
    score = lexical_match_score(
        "What is the grading breakdown?",
        "Course grading: quizzes 20 percent, final project 40 percent, final exam 30 percent, attendance 10 percent.",
    )
    assert score >= 5


def test_citation_validation_requires_exact_filename_and_page():
    hits = [{"page": {"file": "Week 2 Slides.pptx", "page": 33}}]
    supported = main._validate("The meme is shown here [Week 2 Slides.pptx p.33].", hits)
    bullet_supported = main._validate("- Week 2 Slides.pptx p.33 — meme evidence.", hits)
    unsupported = main._validate("The meme is shown here [Week 2 Slides.pptx].", hits)
    assert supported["all_sources_supported"] is True
    assert bullet_supported["all_sources_supported"] is True
    assert unsupported["all_sources_supported"] is False


def test_validation_allows_retrieved_but_uncited_candidates():
    hits = [
        {"page": {"file": "slides.pptx", "page": 10}},
        {"page": {"file": "slides.pptx", "page": 11}},
    ]
    result = main._validate("Answer supported by [slides.pptx p.10].", hits)
    assert result["all_sources_supported"] is True
    assert result["sources_not_cited"] == [{"file": "slides.pptx", "page": 11}]


def test_missing_information_detector_accepts_not_mentioned_anywhere():
    category, question = QUESTIONS[-1]
    row = run_one(
        FakeSessionWithAnswer("This is not mentioned anywhere in the materials."),
        "http://example.test", 9, category, question, MODES[0]
    )
    assert row["acknowledges_missing_information"] is True


def test_empty_retrieval_results_do_not_raise_index_error():
    class EmptyBM25:
        def retrieve(self, *args, **kwargs):
            return type("Result", (), {"documents": [[]], "scores": [[]]})()

    class EmptyEncoder:
        def encode(self, query):
            return type("Vector", (), {"tolist": lambda self: [0.0]})()

    class EmptyTextCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "distances": [[]]}

    corpus = Corpus.__new__(Corpus)
    corpus.chunks = [{
        "cid": "doc__p1__0", "doc": "doc", "file": "notes.txt", "kind": "txt",
        "page": 1, "section": None, "text": "A neutral note.", "image": None,
        "seg": 0, "page_index": 0,
    }]
    corpus.pages = [{"doc": "doc", "file": "notes.txt", "kind": "txt", "page": 1,
                     "section": None, "text": "A neutral note.", "image": None}]
    corpus.retriever = EmptyBM25()
    corpus.text_enc = EmptyEncoder()
    corpus.text_coll = EmptyTextCollection()
    corpus.visual_coll = type("EmptyVisual", (), {"count": lambda self: 0})()

    assert corpus._retrieve_unlocked("unmatched", retrieval_mode="text_keyword_only") == []


def test_api_forwards_default_and_explicit_retrieval_mode(monkeypatch):
    seen = []

    class EmptyCorpus:
        def retrieve(self, question, doc=None, topic=None, retrieval_mode="hybrid"):
            seen.append(retrieval_mode)
            return []

    monkeypatch.setattr(main.hybrid, "get_corpus", lambda: EmptyCorpus())
    main.ask({"question": "test"})
    main.ask({"question": "test", "retrieval_mode": "text_keyword_only"})
    main.ask({"question": "Compare the charts"})
    assert seen == ["text_keyword_only", "text_keyword_only", "hybrid"]


def test_evaluator_row_contains_timing_sources_validation_and_manual_fields():
    category, question = QUESTIONS[-1]
    row = run_one(FakeSession(), "http://example.test", 9, category, question, MODES[0])

    assert row["question_number"] == 9
    assert row["retrieval_mode"] == "hybrid"
    assert row["elapsed_seconds"] >= 0
    assert row["returned_source_count"] == 0
    assert row["validation"] is True
    assert row["acknowledges_missing_information"] is True
    assert row["correct_manual_review"] is None
    assert row["sources_supported_manual_review"] is None


def test_missing_information_detector_accepts_not_stated_anywhere():
    category, question = QUESTIONS[-1]
    row = run_one(
        FakeSessionWithAnswer("The preference is not stated anywhere in the supplied course evidence."),
        "http://example.test", 9, category, question, MODES[0]
    )
    assert row["acknowledges_missing_information"] is True


def test_canonical_evaluation_set_has_nine_questions():
    assert len(QUESTIONS) == 9
    assert QUESTIONS[-1][0].replace("*", "").strip().lower() == "unanswerable"
