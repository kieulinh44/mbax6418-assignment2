"""Project configuration.

Model endpoints are OpenAI-compatible. Values can be overridden with
environment variables (e.g. DOBOLYI_CHAT_KEY) so the API key never needs to
live in the repo. Defaults point at the class's local endpoints.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")      # put course files here
DATA_INDEX = os.path.join(BASE_DIR, "data", "index")  # generated pages + images + vectors

# Chat / reasoning model (grounded answers, quiz writing) — DeepSeek
CHAT_BASE = os.environ.get("DOBOLYI_CHAT_BASE", "http://dobolyi.com:9000/v1")
CHAT_MODEL = os.environ.get("DOBOLYI_CHAT_MODEL", "DeepSeek-V4-Flash-0731")
CHAT_KEY = os.environ.get("DOBOLYI_CHAT_KEY", "6418")

# Vision model (reads diagrams / charts in page images) — Qwen
VISION_BASE = os.environ.get("DOBOLYI_VISION_BASE", "http://dobolyi.com:9001/v1")
VISION_MODEL = os.environ.get("DOBOLYI_VISION_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
VISION_KEY = os.environ.get("DOBOLYI_VISION_KEY", "6418")

# Local embedding model (no embeddings endpoint available remotely)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval
RETRIEVE_TOP_K = 6
