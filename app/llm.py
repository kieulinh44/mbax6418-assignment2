"""OpenAI-compatible client for the local model endpoints.

Handles reasoning models (DeepSeek/Qwen emit a `reasoning` field and spend
tokens there), so we request a generous max_tokens, disable thinking for these
grounded tasks, and accept completed `content` only — never internal reasoning.
Vision requests pass image(s) as base64 data URIs.
"""
import base64
import os
import requests

from . import config


def _complete(base, key, model, messages, max_tokens=4096, temperature=0.2):
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"].get("content")
    # Internal reasoning is not a completed, user-facing answer and must never
    # become evidence for another model or be displayed to the student.
    if not isinstance(content, str) or not content.strip() or choice.get("finish_reason") == "length":
        raise RuntimeError("Model did not return a complete final answer; try again with a larger token budget.")
    return content.strip()


def _image_data_uri(path):
    # Internal native-resolution details are already encoded in memory.
    if path.startswith("data:image/"):
        return path
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def chat(messages, max_tokens=4096, temperature=0.2):
    """Text chat against the reasoning model (DeepSeek)."""
    return _complete(config.CHAT_BASE, config.CHAT_KEY, config.CHAT_MODEL,
                     messages, max_tokens=max_tokens, temperature=temperature)


def ground(messages, max_tokens=4096):
    """Run a second, low-temperature pass that edits an answer for evidence support."""
    return _complete(config.CHAT_BASE, config.CHAT_KEY, config.CHAT_MODEL,
                     messages, max_tokens=max_tokens, temperature=0.0)


def vision(prompt, image_paths, max_tokens=4096):
    """Ask the vision model (Qwen) about one or more page images."""
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": _image_data_uri(p)}})
    return _complete(config.VISION_BASE, config.VISION_KEY, config.VISION_MODEL,
                     [{"role": "user", "content": content}],
                     max_tokens=max_tokens, temperature=0.1)
