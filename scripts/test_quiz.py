"""Quick end-to-end test of quiz generation + grading (run against the local server)."""
import json
import sys

import requests

BASE = "http://127.0.0.1:8000"


def main():
    # generate a quiz from Week 5 / RAG
    r = requests.post(f"{BASE}/api/quiz",
                      json={"doc": "mbax-6418-week-5-context-engineering-and-rag-v1",
                            "topic": "RAG", "n": 2}, timeout=240)
    r.raise_for_status()
    q = r.json()
    print("quiz_id:", q["quiz_id"])
    print("title:", q["title"], "| n:", q["n"])
    for x in q["questions"]:
        leak = [k for k in ("answer_idx", "explanation") if k in x]
        print(f"  Q {x['id']}: {x['question'][:65]}")
        print(f"     opts: {[o[:18] for o in x['options']]}")
        print(f"     key leaked to client? {leak}  | source {x['source']['file']} p{x['source']['page']}")

    print("\n--- grading (correct + wrong on purpose) ---")
    answers = {}
    for i, x in enumerate(q["questions"]):
        answers[x["id"]] = i % 4  # first correct-ish, second maybe wrong
    rr = requests.post(f"{BASE}/api/quiz/grade", json={"quiz_id": q["quiz_id"], "answers": answers}, timeout=60)
    rr.raise_for_status()
    g = rr.json()
    print("score:", g["score"], "/", g["total"], f"({g['percent']}%)")
    for res in g["results"]:
        print(f"  {res['id'].upper()} correct={res['correct']} correct_ans={res['correct_answer']} "
              f"your={res['your_answer']} | explain: {res['explanation'][:60]}")

    print("\n--- reveal (no answers) ---")
    rv = requests.post(f"{BASE}/api/quiz/grade", json={"quiz_id": q["quiz_id"], "answers": {}}, timeout=60)
    rv.raise_for_status()
    print("reveal_only:", rv.json().get("reveal_only"), "| results:", len(rv.json()["results"]))


if __name__ == "__main__":
    main()
