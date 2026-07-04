"""End-to-end dry run: sample WAV -> VAD -> faster-whisper -> cleanup -> print.

Exercises the real pipeline (no mocks) minus the OS-specific pieces, so it
works on any platform. If Ollama isn't running, cleanup falls back to the
rule-based pass and says so.

Usage: python scripts/dry_run.py [path/to/audio.wav]
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audio import load_wav           # noqa: E402
from app.cleanup import clean_transcript  # noqa: E402
from app.config import load_config        # noqa: E402
from app.stt import Transcriber           # noqa: E402


def main():
    wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "sample.wav"
    config_path = ROOT / "config.json"
    config = load_config(config_path if config_path.exists() else None)

    print(f"[1/4] loading audio: {wav_path}")
    audio = load_wav(wav_path, target_rate=config["audio"]["sample_rate"])
    print(f"      {len(audio) / config['audio']['sample_rate']:.1f}s at "
          f"{config['audio']['sample_rate']} Hz")

    print(f"[2/4] transcribing with faster-whisper "
          f"({config['stt']['model']}, vad_filter={config['stt']['vad_filter']})")
    t0 = time.time()
    transcriber = Transcriber(**config["stt"])
    raw = transcriber.transcribe(audio)
    print(f"      raw transcript ({time.time() - t0:.1f}s): {raw!r}")

    n_words = len(raw.split())
    skip = n_words < config["cleanup"]["skip_llm_under_words"]
    print(f"[3/4] cleanup: {n_words} words -> "
          + ("skipping LLM (<10-word rule)" if skip
             else f"Ollama {config['cleanup']['ollama_model']} (rule-based fallback if down)"))
    cleaned = clean_transcript(raw, config)

    print("[4/4] final cleaned text:")
    print(f"      >>> {cleaned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
