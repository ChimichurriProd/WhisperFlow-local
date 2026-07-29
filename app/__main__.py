"""CLI entrypoint: `python -m app` starts the push-to-talk loop (macOS).

Use --menubar for the menu-bar + floating-pill app; without it, a plain
terminal process that logs to stdout.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Local WhisperFlow clone: hold a global hotkey to dictate; "
            "faster-whisper transcribes, Ollama cleans, and the text is "
            "injected at your cursor. Fully offline."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="path to config.json (default: ./config.json if present, else built-ins)",
    )
    parser.add_argument(
        "--menubar",
        action="store_true",
        help="run as a menu-bar app (status icon) instead of a terminal process",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def hold_instance_lock(path=None):
    """Take the single-instance lock, or return None if another engine holds it.

    Two engines at once is a real failure mode we have hit, not a hypothetical:
    the log shows a second instance (launched from a stale duplicate bundle)
    fighting the first — both grab the mic and the hotkey, and whichever saves
    config.json last clobbers the other's settings. flock() is advisory and
    dies with the process, so a crash can never leave a stale lock behind.
    The returned file object must be kept alive for the process lifetime.
    """
    import fcntl

    if path is None:
        path = Path.home() / "Library" / "Logs" / "whisperflow-local" / "engine.lock"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    import os

    f.write(str(os.getpid()))
    f.flush()
    return f


def main(argv=None):
    args = build_parser().parse_args(argv)

    lock = hold_instance_lock()
    if lock is None:
        print("[app] another WhisperFlow engine is already running — exiting "
              "(two engines fight over the mic, the hotkeys and config.json)",
              flush=True)
        return 0
    main._lock = lock  # keep the fd alive for the process lifetime

    config_path = args.config
    if config_path is None:
        # CWD first, then the project root (matters under launchd).
        for candidate in (Path("config.json"),
                          Path(__file__).resolve().parent.parent / "config.json"):
            if candidate.exists():
                config_path = candidate
                break
    config = load_config(config_path)

    # Imports after arg parsing so --help stays instant.
    if args.menubar:
        from .menubar import run_menubar

        run_menubar(config, config_path=str(config_path) if config_path else None)
    else:
        from .hotkey import PushToTalkApp

        PushToTalkApp(config).run()


if __name__ == "__main__":
    sys.exit(main())
