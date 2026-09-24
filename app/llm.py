"""OpenAI-compatible client for the local model endpoints.

Handles reasoning models (DeepSeek/Qwen emit a `reasoning` field and spend
tokens there), so we always request a generous max_tokens and read `content`.
Vision requests pass image(s) as base64 data URIs.
"""
import base64
import os
import requests

from . import config


def _complete(base, key, model, messages, max_tokens=1024, temperature=0.2):
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    # Some reasoning backends return content=None but keep the text in `reasoning`.
    if content is None:
        content = msg.get("reasoning")
    return (content or "").strip()


def _image_data_uri(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def chat(messages, max_tokens=1024, temperature=0.2):
    """Text chat against the reasoning model (DeepSeek)."""
    return _complete(config.CHAT_BASE, config.CHAT_KEY, config.CHAT_MODEL,
                     messages, max_tokens=max_tokens, temperature=temperature)


def vision(prompt, image_paths, max_tokens=1024):
    """Ask the vision model (Qwen) about one or more page images."""
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": _image_data_uri(p)}})
    return _complete(config.VISION_BASE, config.VISION_KEY, config.VISION_MODEL,
                     [{"role": "user", "content": content}],
                     max_tokens=max_tokens, temperature=0.1)
