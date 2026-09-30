"""Quiz answer-key hiding + grading (requirement 3a), fully offline.

Stubs the LLM and the corpus so no model endpoint or index build is needed.
Verifies:
  - /api/quiz-style response never contains answer_idx or explanation
  - grading returns score/verdicts/explanations after answering
  - reveal (no answers) returns solutions on demand
"""
import json

import pytest

from app import quiz

PAGES = [
    {"doc": "deck-a", "file": "deck-a.pdf", "kind": "pdf", "page": 1,
     "section": "Intro", "text": "Overfitting is when a model memorizes the "
                                 "training data instead of learning the pattern.",
     "image": None},
    {"doc": "deck-a", "file": "deck-a.pdf", "kind": "pdf", "page": 2,
     "section": "Regularisation", "text": "Regularisation penalizes complex models.",
     "image": None},
]

MODEL_JSON = [
    {"question": "What is overfitting?", "options": ["A", "B", "C", "D"],
     "answer_idx": 0, "explanation": "The deck defines it as memorizing training data.",
     "source_file": "deck-a.pdf", "source_page": 1},
]


class FakeLLM:
    @staticmethod
    def chat(messages, max_tokens=1024, temperature=0.2):
        return json.dumps(MODEL_JSON)


class FakeCorpus:
    @staticmethod
    def retrieve(query, top_k=None, doc=None, topic=None):
        return [{"page": p, "chunk": p["text"], "score": 1.0} for p in PAGES[:1]]


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr(quiz.llm, "chat", FakeLLM.chat)
    monkeypatch.setattr(quiz.hybrid, "get_corpus", lambda: FakeCorpus())


def test_quiz_response_hides_answer_key(stub_models):
    q = quiz.generate_quiz(question_theme="overfitting", n=1)
    assert q is not None
    assert q["n"] == 1
    public = q["questions"][0]
    assert "answer_idx" not in public, "answer key leaked to the client"
    assert "explanation" not in public, "explanation leaked before answering"
    assert public["question"] and public["options"], "question/options missing"


def test_grade_scoring_and_reveal(stub_models):
    q = quiz.generate_quiz(question_theme="overfitting", n=1)
    qid = q["quiz_id"]
    qkey = q["questions"][0]["id"]

    # correct answer
    graded = quiz.grade_quiz(qid, {qkey: 0})
    assert graded["score"] == 1 and graded["percent"] == 100
    assert graded["results"][0]["correct"] is True
    assert graded["results"][0]["explanation"], "explanation missing after answering"
    assert graded["results"][0]["source"]["file"] == "deck-a.pdf"

    # wrong answer
    graded = quiz.grade_quiz(qid, {qkey: 1})
    assert graded["score"] == 0 and graded["results"][0]["correct"] is False

    # reveal on demand (no answers)
    revealed = quiz.grade_quiz(qid, {})
    assert revealed["reveal_only"] is True
    assert revealed["results"][0]["correct_answer"] == 0
