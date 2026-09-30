"""Project configuration.

Model endpoints are OpenAI-compatible. All connection details come from
environment variables (or a local, git-ignored `.env` file — see
`.env.example`), so no real key or class endpoint ever lives in the
repository. If the variables are unset the app runs with empty values and
model calls will fail loudly — set DOBOLYI_* in `.env` before use.

Security rule (assignment requirement): NEVER commit real keys or the class
endpoint URLs. Keep dummy values in `.env.example` only.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")      # put course files here
DATA_INDEX = os.path.join(BASE_DIR, "data", "index")  # generated pages + images + vectors

# Chat / reasoning model (grounded answers, quiz writing)
CHAT_BASE = os.environ.get("DOBOLYI_CHAT_BASE", "")
CHAT_MODEL = os.environ.get("DOBOLYI_CHAT_MODEL", "")
CHAT_KEY = os.environ.get("DOBOLYI_CHAT_KEY", "")

# Vision model (reads diagrams / charts in page images)
VISION_BASE = os.environ.get("DOBOLYI_VISION_BASE", "")
VISION_MODEL = os.environ.get("DOBOLYI_VISION_MODEL", "")
VISION_KEY = os.environ.get("DOBOLYI_VISION_KEY", "")

# Local embedding model (no embeddings endpoint available remotely)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval
RETRIEVE_TOP_K = 6
