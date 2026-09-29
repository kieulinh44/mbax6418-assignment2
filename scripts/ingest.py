"""CLI: ingest course files from data/raw and build the vector index.

Usage:  python -m scripts.ingest
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, ingest, hybrid  # noqa: E402


def main():
    if not os.path.isdir(config.DATA_RAW):
        os.makedirs(config.DATA_RAW, exist_ok=True)
    print(f"Ingesting from: {config.DATA_RAW}")
    pages = ingest.ingest_dir()
    if not pages:
        print("No supported files found. Put PDFs/pptx/docx/md/txt into data/raw first.")
        return
    print("Building hybrid index (keyword BM25 + text embeddings + visual CLIP images)…")
    hybrid.rebuild_index()
    print(f"Hybrid index ready: {len(pages)} pages in chromadb + bm25s.")


if __name__ == "__main__":
    main()
