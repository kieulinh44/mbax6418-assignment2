#!/usr/bin/env python3
"""Secrets-leak scan over git-tracked files.

Fails (exit 1) when a real credential pattern appears in committed files:
  - the class endpoint domain `dobolyi.com`
  - `sk-...` API-key shapes
  - a non-empty value assigned to a *_KEY setting in Python source
Allows dummy values in `.env.example` and ignores generated/binary paths.
Run: python scripts/scan_secrets.py   (or CI)
"""
from __future__ import annotations

import re
import subprocess
import sys

# Patterns that indicate a real secret / endpoint in committed files.
PATTERNS = [
    (re.compile(r"dobolyi\.com", re.I), "class endpoint domain"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "API key (sk-...)"),
    (re.compile(r"(?:CHAT_KEY|VISION_KEY|API_KEY|OPENAI_API_KEY)\s*=\s*[\"'][^\"']{4,}[\"']"),
     "hard-coded *_KEY assignment"),
]

EXCLUDE_SUFFIXES = (".png", ".jpg", ".jpeg", ".json.gz", ".pyc")
EXCLUDE_PATHS = (".git/", "data/raw/", "data/index/", ".env", ".env.example",
                 "scripts/scan_secrets.py", "docs/screenshots/")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f.strip()]


def main() -> int:
    hits = []
    for path in tracked_files():
        low = path.lower()
        if any(low.startswith(e) for e in EXCLUDE_PATHS) or low.endswith(EXCLUDE_SUFFIXES):
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for rx, label in PATTERNS:
            for m in rx.finditer(text):
                hits.append((path, label, m.group(0)))
    if hits:
        print("SECRET-SCAN FAILED — potential credentials in committed files:")
        for path, label, sample in hits:
            shown = sample if len(sample) <= 24 else sample[:10] + "…" + sample[-6:]
            print(f"  {path}: {label} -> {shown}")
        return 1
    print("Secret scan clean: no real credentials in committed files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
