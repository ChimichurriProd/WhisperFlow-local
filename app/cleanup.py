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


_TOKEN_STRIP = ",.!?;:\"'()…"

# Fused repetition loops: Whisper can weld a repeat loop into ONE token with no
# spaces ("useertasertasertas…" — seen injected into a real document), which
# the token-level collapse below can never catch. A non-space unit of 3-12
# chars repeated 4+ times back-to-back is noise, not language (even Swedish
# compounds like "barnbarnsbarn" only reach 3 in a row).
_FUSED_REPEAT_RE = re.compile(r"([^\s]{3,12}?)\1{3,}")


def collapse_repeats(text):
    """Whisper hallucination guard (seen in the wild: "myślę, " repeated 94
    times on a short Swedish utterance misdetected as English). Whisper keeps
    its final decode even when every temperature fallback fails the quality
    checks, so loops must be caught here, after STT:

    - an utterance of >=5 tokens that are all the SAME token -> "" (drop it)
    - a 2-3-token phrase repeated >=3 times in a row -> one occurrence
    - a single token repeated >=4 times in a row -> one occurrence
      (>=4 so dictated "nej, nej, nej" survives)
    - text with no letters or digits at all (a lone "!") -> ""
    """
    text = (text or "").strip()
    if not text:
        return ""
    text = _FUSED_REPEAT_RE.sub(r"\1", text)  # in-token loops first
    tokens = text.split()
    norm = [t.strip(_TOKEN_STRIP).lower() for t in tokens]
    if len(tokens) >= 5 and len(set(norm)) == 1:
        return ""
    out, i = [], 0
    while i < len(tokens):
        collapsed = False
        for unit in (3, 2, 1):
            min_reps = 4 if unit == 1 else 3
            if i + unit * min_reps > len(tokens):
                continue
            reps = 1
            while norm[i:i + unit] == norm[i + reps * unit:
                                           i + (reps + 1) * unit]:
                reps += 1
            if reps >= min_reps:
                out.extend(tokens[i:i + unit])
                i += reps * unit
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    result = " ".join(out)
    if not re.search(r"[^\W_]", result):  # no letters/digits left -> noise
        return ""
    return result


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
    # STT-artifact guard first — runs even in verbatim mode (a hallucination
    # loop is not "what was said") and before the word count, so 94 junk
    # tokens never route the utterance to the LLM.
    text = collapse_repeats(text)
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
