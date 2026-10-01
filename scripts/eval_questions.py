"""Run the evaluation question set (docs/EVALUATION.md) against the local app.

Usage:  python scripts/eval_questions.py
Requires the app running at http://127.0.0.1:8000 (uvicorn app.main:app).
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("CA_BASE", "http://127.0.0.1:8000")

QUESTIONS = [
    # (category, question)
    ("Syllabus", "What is the grading breakdown for MBAX 6418?"),
    ("Syllabus", "How is attendance and participation evaluated?"),
    ("Slide text (Week 3)", "What makes a good few-shot example in prompt engineering?"),
    ("Slide text (Week 5)", "What is Retrieval Augmented Generation, and why is it useful?"),
    ("Slide text (Week 4)", "Why are commits useful when debugging, according to the Week 4 material?"),
    ("Slide text (Week 2)", "What is vibe coding?"),
    ("VISUAL", "What does the flowchart on the \"What Is RAG?\" slide show?"),
    ("VISUAL (meme)", "Explain the meme on the \"Vibe Coding on Prod\" slide (Week 2): which meme format is it and what point does it make?"),
    ("UNANSWERABLE", "What is the professor's favorite programming language?"),
]


def main():
    r = requests.get(f"{BASE}/api/health", timeout=10)
    r.raise_for_status()
    print(f"App alive at {BASE}\n")
    for i, (cat, q) in enumerate(QUESTIONS, 1):
        print(f"───── Q{i} [{cat}] ─────")
        print("Q:", q)
        try:
            resp = requests.post(f"{BASE}/api/ask", json={"question": q}, timeout=240)
            d = resp.json()
            print("A:", (d.get("answer") or "")[:400])
            sources = d.get("sources", [])
            print(f"Sources ({len(sources)}):")
            for s in sources[:4]:
                img = " [image]" if s.get("image") else ""
                print(f"  · {s['file']} · page/slide {s['page']}{img}")
            v = d.get("validation", {})
            print("Validation: checked", v.get("checked"),
                  "| all_sources_supported", v.get("all_sources_supported"))
        except Exception as e:
            print("ERR:", e)
        print()
    print("Reminder: check the NOTES in docs/EVALUATION.md for expected behavior "
          "per question (esp. the meme slide image and the missing-info answer).")


if __name__ == "__main__":
    main()
