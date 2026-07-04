"""Transcript cleanup: rule-based filler stripping, plus Ollama polish for longer text.

Latency rule: utterances under `skip_llm_under_words` words never touch the LLM —
rule-based cleanup only. Ollama being down degrades gracefully to rules too.
"""

import re

import requests

# Standalone fillers to strip (word-boundary matched, case-insensitive).
# English + Swedish (öh/öhm/asså are the Swedish um/uh/y'know equivalents).
FILLER_WORDS = (
    "um", "uh", "uhm", "erm", "hmm", "mhm", "you know", "i mean", "like,",
    "öh", "öhm", "eh", "ehm", "asså",
)

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b[,.]?\s*",
    re.IGNORECASE,
)

CLEANUP_PROMPT = (
    "You clean up dictated text. Remove filler words and false starts, fix "
    "capitalization and punctuation, and preserve the speaker's meaning and "
    "wording. Keep the text in its original language — Swedish stays Swedish, "
    "English stays English; never translate. Output ONLY the cleaned text "
    "with no commentary.\n\n"
    "Dictated text:\n{text}"
)


def strip_fillers(text):
    """Rule-based cleanup: drop fillers, collapse whitespace, capitalize, punctuate."""
    cleaned = _FILLER_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)  # space before punctuation
    cleaned = re.sub(r"^[,.\s]+", "", cleaned)          # leading orphan punctuation
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
    return cleaned


def ollama_clean(text, ollama_url, ollama_model, timeout=30, keep_alive="30m"):
    """Ask the local Ollama server to polish *text*. Raises on HTTP errors.

    keep_alive keeps the model resident between dictations so it doesn't pay a
    multi-second cold reload each time (Ollama unloads after 5 min by default).
    """
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": ollama_model,
            "prompt": CLEANUP_PROMPT.format(text=text),
            "stream": False,
            "keep_alive": keep_alive,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def clean_transcript(text, config):
    """Full cleanup pipeline honoring the <N-word LLM skip and Ollama fallback."""
    text = text.strip()
    if not text:
        return ""

    cfg = config["cleanup"]
    word_count = len(text.split())

    if word_count < cfg["skip_llm_under_words"]:
        # Latency rule: short utterances skip the LLM entirely.
        return strip_fillers(text)

    try:
        return ollama_clean(
            text,
            ollama_url=cfg["ollama_url"],
            ollama_model=cfg["ollama_model"],
            timeout=cfg["timeout_seconds"],
            keep_alive=cfg.get("keep_alive", "30m"),
        )
    except (requests.RequestException, KeyError, ValueError):
        # Ollama down/misconfigured: degrade to rule-based cleanup, never block.
        return strip_fillers(text)
