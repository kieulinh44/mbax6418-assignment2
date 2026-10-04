"""Unit tests: quiz generation, server-side answer key, grading, reveal."""
import pytest

from app import quiz

FAKE_HITS = [
    {"page": {"doc": "d", "file": "deck.pptx", "page": 10,
              "section": "What Is RAG?", "image": None, "text": "RAG definition text"},
     "chunk": "RAG definition chunk", "score": 1.0},
    {"page": {"doc": "d", "file": "deck.pptx", "page": 18,
              "section": None, "image": None, "text": "Reranking text"},
     "chunk": "Reranking chunk", "score": 0.9},
]

MODEL_JSON = """```json
[
  {"question": "What is RAG?", "options": ["a", "b", "c", "d"],
   "answer_idx": 1, "explanation": "From slide 10.", "source_file": "deck.pptx", "source_page": 10},
  {"question": "What does reranking do?", "options": ["x", "y", "z", "w"],
   "answer_idx": 2, "explanation": "From slide 18.", "source_file": "deck.pptx", "source_page": 18}
]
```"""


class FakeCorpus:
    def retrieve(self, *a, **k):
        return FAKE_HITS


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(quiz.hybrid, "get_corpus", lambda: FakeCorpus())
    monkeypatch.setattr(quiz.llm, "chat",
                        lambda messages, max_tokens=1024: MODEL_JSON)


def test_extract_json_tolerant():
    assert quiz._extract_json("```json\n[1, 2]\n```") == [1, 2]
    assert quiz._extract_json("prose {a: 1}") is None  # not valid JSON
    assert quiz._extract_json("[1, 2]") == [1, 2]


def test_generate_quiz_hides_answer_key(patched):
    q = quiz.generate_quiz(question_theme="RAG", n=2)
    assert q is not None
    assert len(q["questions"]) == 2
    for public in q["questions"]:
        assert "answer_idx" not in public
        assert "explanation" not in public
        assert len(public["options"]) == 4
        assert public["source"]["file"] == "deck.pptx"


def test_generate_quiz_clamps_n(patched):
    q = quiz.generate_quiz(question_theme="RAG", n=99)
    assert 2 <= len(q["questions"]) <= 6


def test_grade_scores_and_explains(patched):
    q = quiz.generate_quiz(question_theme="RAG", n=2)
    res = quiz.grade_quiz(q["quiz_id"], answers={"q1": 1, "q2": 2})
    assert res["score"] == 2 and res["percent"] == 100
    assert all(r["correct"] for r in res["results"])
    assert res["results"][0]["correct_answer"] == 1
    assert "From slide" in res["results"][0]["explanation"]


def test_grade_non_numeric_answer_not_counted(patched):
    q = quiz.generate_quiz(question_theme="RAG", n=2)
    res = quiz.grade_quiz(q["quiz_id"], answers={"q1": "banana"})
    assert res["results"][0]["your_answer"] is None
    assert res["results"][0]["correct"] is False


def test_reveal_without_answers(patched):
    q = quiz.generate_quiz(question_theme="RAG", n=2)
    res = quiz.grade_quiz(q["quiz_id"], answers={})
    assert res["reveal_only"] is True
    assert res["results"][0]["correct_answer"] is not None


def test_unknown_quiz_id():
    assert quiz.grade_quiz("nope") is None


def test_model_failure_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(quiz.hybrid, "get_corpus", lambda: FakeCorpus())

    def boom(*a, **k):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(quiz.llm, "chat", boom)
    with pytest.raises(RuntimeError, match="model service unavailable"):
        quiz.generate_quiz(question_theme="RAG", n=1)
