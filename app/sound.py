"""Subtle audio cues so you know when recording starts and text is inserted.

Plays built-in macOS system sounds via `afplay` (spawned, non-blocking, and
thread-safe — no AppKit main-thread requirement). Silently no-ops if disabled,
off-platform, or the sound file is missing.
"""

import subprocess
import sys
from pathlib import Path

_SOUND_DIR = Path("/System/Library/Sounds")


def _play(name):
    path = _SOUND_DIR / f"{name}.aiff"
    if sys.platform != "darwin" or not path.exists():
        return
    try:
        subprocess.Popen(
            ["afplay", str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # cues are non-essential


def play_start(config):
    """Cue when recording begins."""
    cues = config.get("sound_cues", {})
    if cues.get("enabled", True):
        _play(cues.get("start", "Tink"))


def play_done(config):
    """Cue when transcribed text has been inserted."""
    cues = config.get("sound_cues", {})
    if cues.get("enabled", True):
        _play(cues.get("done", "Pop"))
