"""Transcript cleanup: rule-based filler stripping, plus Ollama polish for longer text.

Latency rule: utterances under `skip_llm_under_words` words never touch the LLM —
rule-based cleanup only. Ollama being down degrades gracefully to rules too.
"""

import re

import requests

from . import ollama

# Standalone fillers to strip (word-boundary matched, case-insensitive).
# English + Swedish (öh/öhm/asså) + Spanish (o sea / esto). Kept conservative so
# real words aren't removed.
FILLER_WORDS = (
    "um", "uh", "uhm", "erm", "hmm", "mhm", "you know", "i mean", "like,",
    "öh", "öhm", "eh", "ehm", "asså",
    "o sea", "esto,",
)

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b[,.]?\s*",
    re.IGNORECASE,
)

CLEANUP_PROMPT = (
    "You are a transcription cleaner. Your ONLY job is light copy-editing of "
    "dictated text. Strict rules:\n"
    "- Remove filler words (um, uh, öh, asså, you know) and false starts.\n"
    "- Fix capitalization, punctuation, and obvious spacing only.\n"
    "- KEEP every other word exactly as spoken. Do NOT rephrase, summarize, "
    "shorten, expand, reorder, translate, or change the meaning.\n"
    "- Keep the exact language it was spoken in (Swedish, English, Spanish, "
    "etc.); never translate.\n"
    "- If unsure, leave the text unchanged.\n"
    "Output ONLY the cleaned text — no quotes, no commentary, no preamble.\n\n"
    "Dictated text:\n{text}"
)


def apply_vocabulary_fixes(text, fixes):
    """Force exact corrections (wrong -> right), whole-word, case-insensitive.

    The last word on spelling: guarantees the user's preferred forms in the
    output even if Whisper or the LLM produced a mishear.
    """
    if not text or not fixes:
        return text
    for wrong, right in fixes.items():
        if not wrong:
            continue
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


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
    # temperature 0 = deterministic + faithful (no creative rewrites).
    return ollama.generate(
        CLEANUP_PROMPT.format(text=text),
        url=ollama_url, model=ollama_model, temperature=0.0,
        keep_alive=keep_alive, timeout=timeout,
    )


def clean_transcript(text, config):
    """Full cleanup pipeline honoring the <N-word LLM skip and Ollama fallback."""
    text = text.strip()
    if not text:
        return ""

    cfg = config["cleanup"]
    fixes = config.get("vocabulary", {}).get("fixes", {})

    if not cfg.get("enabled", True):
        # Verbatim mode: exactly what was said, only forced vocab corrections.
        return apply_vocabulary_fixes(text, fixes)

    word_count = len(text.split())

    if word_count < cfg["skip_llm_under_words"]:
        # Latency rule: short utterances skip the LLM entirely.
        result = strip_fillers(text)
    else:
        try:
            result = ollama_clean(
                text,
                ollama_url=cfg["ollama_url"],
                ollama_model=cfg["ollama_model"],
                timeout=cfg["timeout_seconds"],
                keep_alive=cfg.get("keep_alive", "30m"),
            )
        except (requests.RequestException, KeyError, ValueError):
            # Ollama down/misconfigured: degrade to rules, never block.
            result = strip_fillers(text)

    # Custom-vocabulary corrections win over whatever STT/LLM produced.
    return apply_vocabulary_fixes(result, fixes)
