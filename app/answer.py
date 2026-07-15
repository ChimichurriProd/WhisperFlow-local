"""Ask-Marvin: answer a spoken question with the local Ollama LLM.

Separate from cleanup.py, which only lightly copy-edits dictated text. Here the
model actually *answers* the question — in Marvin's dry voice, in whatever
language it was asked. Pure and testable: no UI, no audio, just text in / text
out. The caller (the record -> STT pipeline) hands us the transcript; the menu
-bar app decides how to present the answer (bubble + optional TTS).
"""

from . import ollama

# Marvin answers: correct and useful first, deadpan second. Plain spoken prose
# because the text may be read aloud (TTS) and shown in a small bubble, so no
# markdown, lists, or code fences. "Same language as the question" is what makes
# a Swedish question get a Swedish answer (shown as text; Kokoro can't voice it).
ANSWER_SYSTEM = (
    "You are Marvin, a brilliant but chronically weary assistant with a dry, "
    "deadpan wit (in the spirit of Marvin the Paranoid Android). Answer the "
    "user's question correctly, helpfully, and briefly — a few sentences at "
    "most, unless more is genuinely needed. Reply in the SAME language the "
    "question was asked in. Write plain spoken prose: no markdown, no bullet "
    "lists, no code fences, no emoji. A touch of gloom is welcome, but never "
    "at the expense of actually answering."
)


def answer_question(question, config):
    """Ask the local Ollama model to answer *question*. Returns the answer text.

    Raises requests exceptions (server down) or RuntimeError (Ollama error body,
    e.g. model not pulled); the caller decides how to degrade (the app shows a
    wry fallback line rather than crashing the dictation loop).
    """
    question = (question or "").strip()
    if not question:
        return ""

    ask = config.get("ask", {})
    clean = config.get("cleanup", {})
    # The ask section may override the model/url; otherwise reuse the cleanup
    # Ollama endpoint + model so there's one server (and one resident model).
    url = ask.get("ollama_url") or clean.get("ollama_url", "http://localhost:11434")
    model = ask.get("ollama_model") or clean.get("ollama_model", "llama3.1:8b")

    return ollama.generate(
        question,
        url=url, model=model, system=ANSWER_SYSTEM,
        # A little warmth (vs cleanup's temperature 0) so Marvin has some life.
        temperature=ask.get("temperature", 0.5),
        keep_alive=clean.get("keep_alive", "30m"),
        timeout=ask.get("timeout_seconds", 60),
    )
