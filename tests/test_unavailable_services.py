"""Resilience when the model endpoints are down (assignment 4a test case).

Today `/api/ask` returns HTTP 502 with raw exception text and quiz returns a
MISLEADING 404 ("No matching course material") when the chat model is simply
unavailable. Requirements: the app must degrade gracefully, explain itself
honestly, and never leak internal exception details.
"""
import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PAGES = [
    {"doc": "deck-a", "file": "deck-a.pdf", "kind": "pdf", "page": 1,
     "section": "Intro",
     "text": "Overfitting is when a model memorizes the training data.",
     "image": None},
]

HITS = [{"page": PAGES[0], "chunk": PAGES[0]["text"], "score": 1.0}]


class FakeCorpus:
    @staticmethod
    def retrieve(query, top_k=None, doc=None, topic=None):
        return HITS


def _boom(*a, **kw):
    raise requests.ConnectionError("simulated connection refused (test)")


# --------------------------------------------------------------------- ask
def test_ask_graceful_when_chat_down(monkeypatch):
    monkeypatch.setattr("app.main.llm.chat", _boom)
    monkeypatch.setattr("app.main.hybrid.get_corpus", lambda: FakeCorpus())
    r = client.post("/api/ask", json={"question": "What is overfitting?"})
    assert r.status_code == 200, "ask must not 502 when the model is down"
    body = r.json()
    assert body.get("service_unavailable") is True
    assert "unavailable" in body["answer"].lower(), "answer must say the service is unavailable"
    assert body["sources"], "retrieved evidence must still be returned"


def test_ask_down_does_not_leak_exception_details(monkeypatch):
    monkeypatch.setattr("app.main.llm.chat", _boom)
    monkeypatch.setattr("app.main.hybrid.get_corpus", lambda: FakeCorpus())
    r = client.post("/api/ask", json={"question": "What is overfitting?"})
    body = r.text
    assert "simulated connection refused" not in body, "internal exception leaked"
    assert "Model call failed" not in body, "old 502 message leaked"


def test_ask_normal_path_still_works(monkeypatch):
    monkeypatch.setattr("app.main.llm.chat",
                        lambda messages, **kw: "Overfitting means memorizing the training data.")
    monkeypatch.setattr("app.main.hybrid.get_corpus", lambda: FakeCorpus())
    r = client.post("/api/ask", json={"question": "What is overfitting?"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("service_unavailable", False) is False
    assert "memorizing" in body["answer"].lower()


# -------------------------------------------------------------------- quiz
def test_quiz_graceful_when_chat_down(monkeypatch):
    monkeypatch.setattr("app.quiz.llm.chat", _boom)
    monkeypatch.setattr("app.quiz.hybrid.get_corpus", lambda: FakeCorpus())
    r = client.post("/api/quiz", json={"n": 2})
    assert r.status_code == 503, "quiz must report the service outage, not a misleading 404"
    assert "unavailable" in r.json()["detail"].lower()
    assert "No matching course material" not in r.json()["detail"], \
        "must not blame the selection when the model is down"


def test_quiz_graceful_when_model_output_unparseable(monkeypatch):
    monkeypatch.setattr("app.quiz.llm.chat", lambda messages, **kw: "lorem ipsum not json")
    monkeypatch.setattr("app.quiz.hybrid.get_corpus", lambda: FakeCorpus())
    r = client.post("/api/quiz", json={"n": 2})
    assert r.status_code == 503, "unparseable model output must not 404"
