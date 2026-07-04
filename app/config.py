"""Load and validate config.json, with sane defaults for every key."""

import copy
import json
from pathlib import Path

DEFAULTS = {
    "stt": {
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
        "language": None,  # auto-detect per utterance (e.g. Swedish/English)
        "vad_filter": True,
    },
    "cleanup": {
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3.1:8b",
        "skip_llm_under_words": 10,
        "timeout_seconds": 30,
        "keep_alive": "30m",  # keep the LLM resident to avoid cold-load stalls
    },
    "hotkey": {
        "push_to_talk": "control + shift + space",
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
}

VALID_DELIVERY_METHODS = ("clipboard", "type")


def load_config(path=None):
    """Return DEFAULTS deep-merged with the JSON file at *path* (if given)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for section, values in data.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    method = cfg["injection"]["delivery_method"]
    if method not in VALID_DELIVERY_METHODS:
        raise ValueError(
            f"injection.delivery_method must be one of {VALID_DELIVERY_METHODS}, got {method!r}"
        )
    return cfg
