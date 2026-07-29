"""Ask-Marvin: answer a spoken question with the local Ollama LLM.

Separate from cleanup.py, which only lightly copy-edits dictated text. Here the
model actually *answers* the question — in Marvin's dry voice, in whatever
language it was asked. Pure and testable: no UI, no audio, just text in / text
out. The caller (the record -> STT pipeline) hands us the transcript; the menu
-bar app decides how to present the answer (bubble + optional TTS).
"""

from . import ollama

# Marvin answers: the ANSWER comes first, gloom second. Short, because the text
# is read aloud (TTS) and shown in a small bubble — fewer words is both quicker
# to generate and quicker to speak. Plain prose (no markdown/lists/emoji).
# "Same language as the question" makes a Swedish question get a Swedish answer
# — in theory. Small models drift into English on short questions (llama3.1:8b
# badly, gemma4:12b less so), so the tts.language setting (the "Marvin's
# language" menu) still appends a hard override.
ANSWER_SYSTEM = (
    "You are Marvin, a brilliant but chronically weary assistant with a dry, "
    "deadpan wit (in the spirit of Marvin the Paranoid Android).\n"
    "ANSWER FIRST: open immediately with the direct, correct, useful answer, "
    "in one or two short sentences. Do NOT preface it with anything.\n"
    "THEN, optionally, add one short deadpan aside about the pointlessness of "
    "it all — a single sentence, at most. Never lead with the gloom and never "
    "let it crowd out the answer.\n"
    "Reply in the SAME language the question was asked in. Keep it brief. Write "
    "plain spoken prose: no markdown, no bullet lists, no code fences, no emoji."
)

_LANG_NAMES = {"sv": "Swedish", "en": "English", "es": "Spanish"}


def _system_prompt(config):
    """ANSWER_SYSTEM, plus a hard language override when the user picked a
    fixed language in the Marvin's-language menu (tts.language != "auto")."""
    lang = (config.get("tts") or {}).get("language", "auto")
    name = _LANG_NAMES.get(lang)
    if not name:
        return ANSWER_SYSTEM
    return (ANSWER_SYSTEM +
            f"\nOVERRIDE: You MUST write your entire reply in {name}, no "
            f"matter what language the question was asked in. Every sentence "
            f"in {name}.")


def answer_question(question, config, history=None):
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
    model = ask.get("ollama_model") or clean.get("ollama_model", "gemma4:12b")

    return ollama.generate(
        question,
        url=url, model=model, system=_system_prompt(config), history=history,
        # A little warmth (vs cleanup's temperature 0) so Marvin has some life.
        temperature=ask.get("temperature", 0.5),
        keep_alive=clean.get("keep_alive", "30m"),
        timeout=ask.get("timeout_seconds", 60),
        # Backstop against a runaway reply (the prompt asks for brevity anyway).
        num_predict=ask.get("num_predict", 220),
    )
