import json
from pathlib import Path

import pytest

from app import main
from app.hybrid import normalize_retrieval_mode
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


def test_retrieval_modes_are_validated_and_defaulted():
    assert normalize_retrieval_mode(None) == "hybrid"
    assert normalize_retrieval_mode("text_keyword_only") == "text_keyword_only"
    with pytest.raises(ValueError):
        normalize_retrieval_mode("visual_only")


def test_api_forwards_default_and_explicit_retrieval_mode(monkeypatch):
    seen = []

    class EmptyCorpus:
        def retrieve(self, question, doc=None, topic=None, retrieval_mode="hybrid"):
            seen.append(retrieval_mode)
            return []

    monkeypatch.setattr(main.hybrid, "get_corpus", lambda: EmptyCorpus())
    main.ask({"question": "test"})
    main.ask({"question": "test", "retrieval_mode": "text_keyword_only"})
    assert seen == ["hybrid", "text_keyword_only"]


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


def test_canonical_evaluation_set_has_nine_questions():
    assert len(QUESTIONS) == 9
    assert QUESTIONS[-1][0].replace("*", "").strip().lower() == "unanswerable"
