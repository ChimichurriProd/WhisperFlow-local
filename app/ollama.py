"""Thin client for the local Ollama /api/generate endpoint.

Shared by the two LLM paths so the request shape lives in one place: dictation
cleanup (cleanup.py) and ask-Marvin answers (answer.py).
"""

import requests


def generate(prompt, *, url, model, system=None, temperature=0.0,
             keep_alive="30m", timeout=30):
    """Run a single-shot generation and return the response text (possibly "").

    Raises the usual requests exceptions on transport errors, and RuntimeError
    on an Ollama error body (e.g. a model that isn't pulled), which returns
    HTTP 200 — so callers can tell "server down" from "server said no".
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate", json=payload, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")
    return (data.get("response") or "").strip()
