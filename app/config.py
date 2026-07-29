"""Load and validate config.json, with sane defaults for every key."""

import copy
import json
from pathlib import Path

DEFAULTS = {
    "stt": {
        "engine": "mlx",              # "mlx" (GPU, fast) | "faster-whisper" (CPU)
        "model": "large-v3-turbo",    # best accuracy; ~0.2s on Apple Silicon GPU
        "language": None,             # auto-detect per utterance (Swedish/English)
        # KB-Whisper (50 000 h of Swedish) handles Swedish instead of the model
        # above: "kb-large" | "kb-small" | None to switch it off. With the
        # language on auto-detect this costs a second decode on Swedish
        # utterances only — see stt.Transcriber._transcribe_mlx. MLX only.
        "swedish_model": "kb-large",
        "vad_filter": True,           # faster-whisper only
        "device": "cpu",              # faster-whisper only
        "compute_type": "int8",       # faster-whisper only
    },
    "cleanup": {
        "enabled": True,       # False = verbatim (no AI cleanup)
        "ollama_url": "http://localhost:11434",
        # gemma4:12b — best small-model adherence to non-English instructions
        # (llama3.1:8b drifted into English). Reaches it via ollama.py's
        # /api/chat + think:false; /api/generate returns "" for this model.
        "ollama_model": "gemma4:12b",
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
        "voice": True,                  # speak answers aloud (see "tts")
        "timeout_seconds": 60,          # answers can be longer than cleanup
        "temperature": 0.5,             # a little warmth vs cleanup's 0.0
        # Token cap. Also a LATENCY control: gemma4 emits ~40 tokens/s, so a
        # long reply is dead air before Marvin starts. The prompt asks for
        # 1-2 sentences; this keeps a runaway one from dragging.
        "num_predict": 140,
        # Conversation: prior turns are replayed to the LLM so follow-ups
        # ("and Denmark?") resolve. Kept short — every turn is tokens gemma4
        # re-reads before it can start answering, which is latency you hear.
        "conversation_turns": 3,
        "conversation_idle_seconds": 180,
    },
    "wakeword": {
        # Hands-free "Hey Marvin": say it and he opens an ask episode — no keys.
        # OFF by default. The mic is already open for the app's lifetime (see
        # audio.py); what this adds is one small ONNX inference per 80ms of
        # audio, on the same stream, for as long as the app runs.
        "enabled": False,
        "phrase": "Hey Marvin",   # what the bundled model was trained on
        "model": None,            # path to a custom .onnx; None = the bundled one
        # Score to accept. 0.5 is the model's own calibrated point, but
        # measured near-misses ("hey martin", "hey marvel") reached 0.73-0.79
        # on an out-of-distribution voice, so this starts stricter. A missed
        # wake word costs a repeat; a false one starts recording mid-meeting.
        "threshold": 0.7,
        "debounce": 2.0,          # seconds deaf after a detection
        # Hands-free questions have no key to release, so they end on silence.
        # The countdown starts when SPEECH does, never at the beep — otherwise
        # a pause for thought ends the recording before a word is said.
        "silence_seconds": 0.9,   # this much quiet AFTER you start ends it
        # Minimum loudness counting as speech. The live threshold is the
        # larger of this and 3x the measured room noise, so a quiet mic (low
        # macOS input volume) still registers.
        "silence_rms": 0.004,
        "start_timeout_seconds": 4.0,  # give up if nothing is said at all
        "max_seconds": 15.0,      # hard cap on one hands-free question
        # Log a 5s heartbeat (hops/rms/peak/suppressed) + near misses, so a
        # detector that silently does nothing can be told apart from one
        # that is simply not being fed audio.
        "debug": False,
        # >0 saves that many seconds of the audio the detector scored to
        # ~/Library/Logs/whisperflow-local/wake_dump.wav, so a wake word
        # that will not fire can be diagnosed from what it really heard.
        "dump_seconds": 0,
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
    "tts": {
        # Marvin's voice engines (app/tts.py). "auto" keeps English lines on
        # Kokoro (instant, prerendered quips) and routes everything else to
        # Chatterbox Multilingual (Swedish + 22 more languages, MLX GPU,
        # ~1.4GB lazy-loaded on first non-English line). "kokoro" = the old
        # English-only behaviour; "chatterbox" = every line through Chatterbox.
        "engine": "auto",
        # Language Marvin answers and speaks in ("Marvin's language" menu):
        # "auto" = mirror the question; "sv"/"en"/"es" = forced (both the
        # Ollama answer and the TTS route). Quips stay English regardless.
        "language": "auto",
        "chatterbox_model": "theoracleguy/Chatterbox-Multilingual-MLX-v2-fp16",
        # Path to a ~10s WAV to clone Marvin's voice from (None = the model's
        # built-in voice). See assets/marvin/voice_ref.wav.
        "voice_ref": None,
        "exaggeration": 0.3,  # chatterbox emotion 0-1 (higher = more drama)
        "pitch": 1.35,        # cartoon pitch multiplier (both engines)
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
