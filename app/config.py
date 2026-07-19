"""Load and validate config.json, with sane defaults for every key."""

import copy
import json
from pathlib import Path

DEFAULTS = {
    "stt": {
        "engine": "mlx",              # "mlx" (GPU, fast) | "faster-whisper" (CPU)
        "model": "large-v3-turbo",    # best accuracy; ~0.2s on Apple Silicon GPU
        "language": None,             # auto-detect per utterance (Swedish/English)
        "vad_filter": True,           # faster-whisper only
        "device": "cpu",              # faster-whisper only
        "compute_type": "int8",       # faster-whisper only
    },
    "cleanup": {
        "enabled": True,       # False = verbatim (no AI cleanup)
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3.1:8b",
        "skip_llm_under_words": 10,
        "timeout_seconds": 30,
        "keep_alive": "30m",  # keep the LLM resident to avoid cold-load stalls
    },
    "hotkey": {
        "push_to_talk": "control + shift + space",
        "ask": "control + shift + a",  # hold to ask Marvin a question (see "ask")
        "mode": "hold",  # "hold" = push-to-talk | "toggle" = tap on/off
    },
    "ask": {
        # Hold the ask hotkey, speak a question -> the local LLM answers in a
        # bubble (and aloud when "voice" is on and the answer is English).
        "enabled": True,
        # None -> reuse cleanup.ollama_model, so both paths keep ONE model
        # resident (a distinct model would make Ollama cold-reload on each
        # dictation<->ask switch). Set a string here only to override.
        "ollama_model": None,
        "voice": True,                  # speak English answers (Kokoro TTS)
        "timeout_seconds": 60,          # answers can be longer than cleanup
        "temperature": 0.5,             # a little warmth vs cleanup's 0.0
        "num_predict": 220,             # token cap: a backstop on reply length
    },
    "injection": {
        "delivery_method": "clipboard",  # "clipboard" | "type"
        "restore_clipboard": True,
        "type_char_delay_ms": 5,
        "append_trailing_space": True,  # so back-to-back dictations don't collide
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "mic_gain": 4.5,  # waveform sensitivity; higher = more reactive
    },
    "vocabulary": {
        # 'terms' bias Whisper toward these spellings (names, jargon, brands).
        "terms": ["Ollama", "WhisperFlow", "faster-whisper", "CTranslate2"],
        # 'fixes' force exact corrections after transcription (wrong -> right),
        # whole-word and case-insensitive, for stubborn mishears.
        "fixes": {},
    },
    "sound_cues": {
        "enabled": True,
        "start": "Tink",  # any name from /System/Library/Sounds
        "done": "Pop",
    },
    "ui": {
        # "marvin" (baked clips) | "waveform"
        "pill_style": "marvin",
        # Which Marvin: "B" (refined 'stoned', the default set in assets/marvin),
        # "A" (faithful refresh) or "G" (knitted plush) — alternates live in
        # assets/marvin/_skins/<name>/.
        "marvin_skin": "B",
        # Double-click Marvin to hear a random deadpan quip (local TTS).
        "double_click_talk": True,
    },
}

VALID_DELIVERY_METHODS = ("clipboard", "type")


def load_config(path=None):
    """Return DEFAULTS deep-merged with the JSON file at *path* (if given)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # A corrupt/unreadable config (e.g. the app was killed mid-save)
            # must not stop the app from starting: run on defaults instead.
            print(f"[config] could not read {path} ({exc}); using defaults",
                  flush=True)
            data = {}
        for section, values in data.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    # Migration: the procedural "marvin_live" style was removed — old saved
    # configs fall back to the baked-clips Marvin.
    if cfg.get("ui", {}).get("pill_style") == "marvin_live":
        cfg["ui"]["pill_style"] = "marvin"
    # Always boot on the biggest STT model: clicking Marvin cycles (and
    # persists) the model, so a stray click would otherwise carry a smaller
    # one into the next launch. In-session switching still works.
    cfg["stt"]["model"] = DEFAULTS["stt"]["model"]
    method = cfg["injection"]["delivery_method"]
    if method not in VALID_DELIVERY_METHODS:
        raise ValueError(
            f"injection.delivery_method must be one of {VALID_DELIVERY_METHODS}, got {method!r}"
        )
    return cfg
