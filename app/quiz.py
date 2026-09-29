"""Practice-quiz generation + grading.

The generated quiz keeps its answer key and explanations SERVER-SIDE (keyed by
quiz_id) so solutions are never sent to the client up front. Grading returns a
score + per-question verdicts + explanations only after the student answers;
a reveal returns solutions on demand.
"""
import json
import os
import re
import uuid

from . import config, llm, hybrid

# in-memory quiz store: quiz_id -> {questions:[{...public, answer_idx, explanation}]}
_STORE = {}

LETTERS = "ABCDEFGH"


def _extract_json(text):
    """Tolerant extraction of a JSON array/object from a model response."""
    if not text:
        return None
    text = text.strip()
    # strip code fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        pass
    # try first { ... last } or [ ... last ]
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return None


def _source_for(page):
    return {
        "file": page["file"],
        "page": page["page"],
        "section": page.get("section"),
        "excerpt": (page.get("text") or "")[:400],
        "image": f"/pages/{os.path.basename(page['image'])}" if page.get("image") else None,
    }


def _build_lookup(pages):
    look = {}
    for p in pages:
        look[(p["file"], p["page"])] = p
    return look


def generate_quiz(question_theme=None, n=4, doc=None, topic=None, max_tokens=3000):
    """Retrieve candidate pages, have the model write MCQs, return quiz with a
    fixed answer key (answer_idx + explanation) in server-side storage."""
    theme = (question_theme or "").strip() or "the selected course material"
    pages_res = hybrid.get_corpus().retrieve(theme, top_k=10, doc=doc, topic=topic)
    if not pages_res:
        return None
    hits = [r["page"] for r in pages_res]

    context = "\n\n".join(
        f"[Document: {p['file']} | page/slide {p['page']}"
        + (f" | section: {p['section']}" if p.get("section") else "")
        + "]\n" + (p.get("text") or "").strip()
        for p in hits[:8]
    )

    sys = (
        "You write multiple-choice PRACTICE quiz questions for a course, grounded "
        "ONLY in the provided course material. Produce a JSON array of exactly "
        f"{n} questions. Each item: {{\"question\": str, \"options\": [exactly 4 short "
        "strings], \"answer_idx\": int 0-3 (index of the correct option), "
        "\"explanation\": one concise sentence grounded in the material, \"source_file\": str, "
        "\"source_page\": int}}. Every question MUST be answerable from the provided "
        "material; do not invent facts. Distractors should be plausible but wrong. "
        "Return ONLY the JSON array — no prose, no markdown fences."
    )
    user = ("Generate a practice quiz about: " + theme + "\n\nCOURSE MATERIAL:\n" + context)
    raw = llm.chat([{"role": "system", "content": sys},
                    {"role": "user", "content": user}], max_tokens=max_tokens)
    data = _extract_json(raw)
    questions = []
    look = _build_lookup(hits)
    n = max(2, min(n, 6))
    if not isinstance(data, list):
        data = []
    for i, q in enumerate(data[:n]):
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 4:
            continue
        opts = [str(o).strip() for o in opts[:4]]
        txt = str(q.get("question", "")).strip()
        if not txt:
            continue
        ans = int(q.get("answer_idx", 0)) % 4
        expl = str(q.get("explanation", "")).strip()
        # map back to a real source page
        src = look.get((q.get("source_file"), int(q.get("source_page", 0))))
        source = _source_for(src) if src else _source_for(hits[0])
        questions.append({
            "id": f"q{i + 1}",
            "question": txt,
            "options": opts,
            "answer_idx": ans,          # FIXED key (server-side only)
            "explanation": expl,
            "source": source,
        })

    if not questions:
        return None

    quiz_id = uuid.uuid4().hex[:12]
    _STORE[quiz_id] = {"questions": questions}
    title_parts = []
    if doc:
        title_parts.append(doc)
    if topic:
        title_parts.append(f"topic: {topic}")
    return {
        "quiz_id": quiz_id,
        "title": "Practice quiz — " + (" / ".join(title_parts) or theme),
        "n": len(questions),
        "doc": doc,
        "topic": topic,
        # NOTE: answer_idx and explanation are intentionally NOT included here.
        "questions": [{k: v for k, v in q.items() if k not in ("answer_idx", "explanation")}
                      for q in questions],
    }


def grade_quiz(quiz_id, answers=None):
    """Return score + per-question verdicts/explanations after answering.
    With no answers (reveal), return the solutions without a score path."""
    quiz = _STORE.get(quiz_id)
    if not quiz:
        return None
    answered = answers or {}
    results = []
    for q in quiz["questions"]:
        selected = answered.get(q["id"])
        correct = selected is not None and int(selected) == q["answer_idx"]
        results.append({
            "id": q["id"],
            "correct_answer": q["answer_idx"],
            "your_answer": int(selected) if selected is not None else None,
            "correct": correct,
            "explanation": q["explanation"],
            "source": q["source"],
        })
    if answered:
        score = sum(1 for r in results if r["correct"])
        return {"quiz_id": quiz_id, "revealed": True, "score": score, "total": len(results),
                "percent": round(100 * score / max(len(results), 1)), "results": results}
    return {"quiz_id": quiz_id, "revealed": True, "reveal_only": True, "total": len(results),
            "results": results}
