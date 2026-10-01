"""Compare retrieval modes on the canonical nine-question evaluation set.

Usage:
    python scripts/compare_retrieval.py

The app must be running and all required course files must be ingested first.
Correctness and source support remain manual-review fields and are never
inferred by this script.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from scripts.eval_questions import QUESTIONS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "evaluation_results.json"
MODES = ("hybrid", "text_keyword_only")


def _is_missing_acknowledged(answer: str, category: str) -> bool:
    if "unanswerable" not in category.lower().replace("*", ""):
        return False
    answer = answer.lower()
    return any(marker in answer for marker in (
        "not in the materials", "information is missing", "do not contain", "not provided",
    ))


def run_one(session: requests.Session, base_url: str, number: int, category: str,
            question: str, mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    payload: dict[str, Any] = {}
    try:
        response = session.post(
            f"{base_url.rstrip('/')}/api/ask",
            json={"question": question, "retrieval_mode": mode},
            timeout=300,
        )
        payload = response.json()
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    elapsed = time.perf_counter() - started
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    validation = payload.get("validation", {}) if isinstance(payload, dict) else {}
    return {
        "question_number": number,
        "category": category,
        "question": question,
        "retrieval_mode": mode,
        "elapsed_seconds": round(elapsed, 6),
        "validation": validation.get("all_sources_supported"),
        "sources": sources,
        "returned_source_count": len(sources) if isinstance(sources, list) else 0,
        "answer": answer,
        "acknowledges_missing_information": _is_missing_acknowledged(answer, category),
        "correct_manual_review": None,
        "sources_supported_manual_review": None,
        "notes": "Manual review required; no automatic correctness claim.",
        "error": error,
    }


def run_comparison(base_url: str = "http://127.0.0.1:8000", output: Path = OUTPUT) -> dict[str, Any]:
    session = requests.Session()
    health = session.get(f"{base_url.rstrip('/')}/api/health", timeout=15)
    health.raise_for_status()
    rows = []
    for number, (category, question) in enumerate(QUESTIONS, start=1):
        for mode in MODES:
            rows.append(run_one(session, base_url, number, category, question, mode))
    result = {
        "evaluation_source": "scripts/eval_questions.py and docs/EVALUATION.md",
        "question_count": len(QUESTIONS),
        "retrieval_modes": list(MODES),
        "manual_review_required": True,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def print_table(result: dict[str, Any]) -> None:
    print("question | category | approach | elapsed_s | validation | sources | answer")
    print("--- | --- | --- | ---: | --- | ---: | ---")
    for row in result["results"]:
        answer = " ".join((row["answer"] or row["error"] or "").split())[:160]
        print(
            f"{row['question_number']} | {row['category']} | {row['retrieval_mode']} | "
            f"{row['elapsed_seconds']:.3f} | {row['validation']} | "
            f"{row['returned_source_count']} | {answer}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run_comparison(args.base_url, args.output)
    print_table(result)
    print(f"\nSaved raw results to {args.output}")
    print("Question 9 missing-information flags and all correctness/source-support fields require human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
