"""Thin client for the local Ollama /api/chat endpoint.

Shared by the two LLM paths so the request shape lives in one place: dictation
cleanup (cleanup.py) and ask-Marvin answers (answer.py).

Why /api/chat and not /api/generate: reasoning-capable models (gemma4, qwen3)
spend their num_predict budget on hidden thinking tokens under /api/generate
and hand back an EMPTY response. /api/chat takes `think: false`, which turns
that off — measured on gemma4:12b, which answers normally here and returns ""
from /api/generate. Non-reasoning models (llama3.1) ignore the flag, so one
code path serves both.
"""

import requests


def generate(prompt, *, url, model, system=None, temperature=0.0,
             keep_alive="30m", timeout=30, num_predict=None, history=None):
    """Run a single-shot chat turn and return the response text (possibly "").

    num_predict caps the tokens generated (a backstop against a runaway reply;
    None = the model's default). Raises the usual requests exceptions on
    transport errors, and RuntimeError whenever Ollama answers with an error
    body (e.g. a model that isn't pulled — 404 here, where /api/generate used
    to say 200) — so callers can tell "server down" from "server said no".
    """
    options = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    # Prior turns, oldest first, so follow-ups ("and Denmark?") resolve against
    # what was actually said. This is the whole reason /api/chat beats a
    # stitched-together single prompt.
    for role, content in (history or []):
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "think": False,  # no hidden reasoning; see the module docstring
        "options": options,
    }
    resp = requests.post(
        f"{url.rstrip('/')}/api/chat", json=payload, timeout=timeout
    )
    # An Ollama complaint (unknown model, bad options) comes back as a JSON
    # error body — read it BEFORE raise_for_status so the caller gets the
    # reason rather than a bare "404 Client Error".
    try:
        data = resp.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")
    resp.raise_for_status()
    if not isinstance(data, dict):
        raise RuntimeError("Ollama returned a non-JSON response")
    return ((data.get("message") or {}).get("content") or "").strip()


def warm(url, model, keep_alive="30m", timeout=30):
    """Preload *model* into Ollama's memory without generating anything (an
    empty-messages request just loads it). Best-effort and silent — it's a
    latency optimization (overlap the load with the user still speaking), not
    required.
    """
    try:
        requests.post(
            f"{url.rstrip('/')}/api/chat",
            json={"model": model, "messages": [], "keep_alive": keep_alive},
            timeout=timeout,
        )
    except Exception:
        pass
