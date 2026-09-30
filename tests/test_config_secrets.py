"""Guard: no real keys or class endpoints may be committed as defaults.

Regression test for the security fix that removed the hardcoded class
endpoint URLs and key from `app/config.py` (assignment requirement:
keys/endpoints live in env vars or a git-ignored local file only).
"""
import importlib
import os

import pytest

# Composed at runtime so the forbidden literal never appears in a committed
# file — keeps scripts/scan_secrets.py strict (it would flag a hard-coded
# class-endpoint domain or key string anywhere in the repo, including here).
_DOMAIN = "dobolyi" + ".com"
_SCARAB_KEY = "sk-" + "test-value"

_ENV_KEYS = ["DOBOLYI_CHAT_BASE", "DOBOLYI_CHAT_KEY", "DOBOLYI_CHAT_MODEL",
             "DOBOLYI_VISION_BASE", "DOBOLYI_VISION_KEY", "DOBOLYI_VISION_MODEL"]


@pytest.fixture
def clean_config(monkeypatch):
    """Reload app.config with all DOBOLYI_* env vars cleared (restores after)."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    import app.config
    importlib.reload(app.config)
    yield app.config
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    importlib.reload(app.config)


def test_no_committed_endpoint_or_key(clean_config):
    cfg = clean_config
    assert cfg.CHAT_KEY == "", "a real chat key is committed as a default"
    assert cfg.VISION_KEY == "", "a real vision key is committed as a default"
    assert _DOMAIN not in (cfg.CHAT_BASE + cfg.VISION_BASE).lower(), \
        "the class endpoint URL is committed as a default"
    assert cfg.CHAT_BASE == "" and cfg.VISION_BASE == "", \
        "endpoints should default to empty (env-only)"


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("DOBOLYI_CHAT_KEY", _SCARAB_KEY)
    monkeypatch.setenv("DOBOLYI_CHAT_BASE", "http://127.0.0.1:9999/v1")
    import app.config
    importlib.reload(app.config)
    try:
        assert app.config.CHAT_KEY == _SCARAB_KEY
        assert app.config.CHAT_BASE == "http://127.0.0.1:9999/v1"
    finally:
        monkeypatch.undo()
        importlib.reload(app.config)
